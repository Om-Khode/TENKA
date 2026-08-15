# assistant/io/api/routes/pairing.py
"""Minting a pair code, and redeeming one.

`POST /v1/pair` is the only unauthenticated write in this API. From Milestone
6b it is reachable from the open internet, and it sits beside a `POST /v1/chat`
that reaches `code_executor` -- so what this module refuses matters more than
what it serves. Four properties carry that weight, and each one has a test:

- **Enrollment is loopback-only.** `POST /v1/pair/code` is gated on
  `require_admin(SYSTEM_CONTROL)`: a remote listener is never `policy.admin`,
  so a compromised phone cannot mint itself a second, wider credential, and a
  loopback device paired only to watch cannot either.
- **Grants ride on the code.** They are chosen on the laptop before the QR is
  drawn and stored on the `PairCode`; the redeeming request never supplies
  them. That is what turns the checkbox row into a boundary rather than a
  suggestion.
- **Wrong, expired and already-used codes are one response.** `consume()`
  already collapses all three to `None`; this module collapses them to one
  `_UNAUTHORIZED` after the same work, exactly as `verify()` does for
  unknown-vs-revoked tokens.
- **No log line ever carries a code.** Not at DEBUG, not in an audit entry.
  The code appears in exactly one place, the response body that the laptop
  renders, and in the QR's URL *fragment* -- which a browser never sends to a
  server, so it cannot land in an intermediary's access log either.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from ..pairing import PairCodeStore
from ..payloads import PairCodePayload
from ..qr import qr_svg
from ..schemas import Envelope, PairCodeRequest, PairRequest
from ..security import (
    _UNAUTHORIZED,
    COOKIE_NAME,
    AuthState,
    accepting_port,
    cookie_kwargs,
    policy_for_scope,
    refuse_unknown_origin,
    require_admin,
)
from ..vault import Capability, Device

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── the pair budget ─────────────────────────────────────────────────────
# One key for the whole route, spent by every caller together. Per-IP keying
# would be worse than useless here: `cloudflared` and `tailscale funnel` both
# connect to this daemon from 127.0.0.1, so every remote caller on earth
# already shares one source key -- and the only per-caller signal left, a
# forwarded-IP header, is the attacker's own input, which would hand out a
# fresh budget per spoofed value. A fixed key is the honest description of
# what this route can actually distinguish, which is nothing.
#
# The accepted cost is that an attacker can deny *her* a pairing window. That
# is a real denial of service and it is taken on purpose: the alternative
# trades a security property (a code cannot be ground down) for an
# availability one, and the remedy costs one command -- minting a fresh code
# clears the lockout (see `mint_pair_code` below), and minting is reachable
# only from the machine itself.
_PAIR_BUDGET_KEY = "pair"

# A ceiling on attempts per minute regardless of whether they fail. There is
# at most one live code, so honest traffic on this route is a handful of
# requests per pairing; 30 is far above that and far below anything worth
# spending CPU on. The *failure* side is what actually bites, and it is the
# shared limiter's own escalating lockout (`RateLimiter.record_failure`,
# 10 consecutive failures) rather than a second threshold defined here -- one
# threshold means the code burns at exactly the moment the route starts
# refusing, with no window in which guessing is throttled but the code still
# stands.
_PAIR_MAX_PER_WINDOW = 30
_PAIR_WINDOW_SECONDS = 60.0


def _store(request: Request) -> PairCodeStore:
    return request.app.state.pair_store


def _endpoints(request: Request) -> list[str]:
    """Every base URL a phone could be told to reach this daemon at.

    6a: the loopback origin alone, built from the port the connection was
    accepted on -- the one piece of addressing a client cannot choose. A
    transport publishes its public hostname onto `app.state.published_hosts`
    in 6b, and that is where it joins this list; it is deliberately not read
    here yet, because an endpoint nobody has tested a QR against is a
    scannable code with no destination.
    """
    port = accepting_port(request.scope)
    if port is None:
        # Unreachable: `authenticate()` resolves policy from this same port
        # and refuses when it cannot. Guarded anyway rather than formatting
        # `None` into a URL a QR would then encode.
        raise _UNAUTHORIZED
    return [f"http://127.0.0.1:{port}"]


# ─── minting: loopback only ──────────────────────────────────────────────
@router.post("/pair/code")
async def mint_pair_code(
    body: PairCodeRequest, request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[PairCodePayload]:
    """Put a code, and a QR for it, in front of the person at the keyboard.

    Minting also clears the pair route's failure lockout. That is what makes
    the accepted denial of service above tolerable rather than permanent: an
    attacker grinding at a code can lock the pairing window shut, and the one
    person who can reopen it is somebody standing at this machine holding
    SYSTEM_CONTROL -- precisely the person being denied. It is not a way in
    for the attacker, because reaching this line at all already required
    everything the attacker does not have.
    """
    try:
        grants = frozenset(Capability(name) for name in body.grants)
    except ValueError:
        # Never echoes the submitted name. `app.py`'s validation handler
        # already strips Pydantic's `input`/`msg` from every 422 body app-wide;
        # this one is built by hand, so it has to keep that promise itself.
        # The literal, not `status.HTTP_422_*`: Starlette renamed that constant
        # and deprecated the old spelling, and this is not a number that moves.
        raise HTTPException(status_code=422, detail="unknown capability")

    store = _store(request)
    # `mint` refuses an empty grant set, but `PairCodeRequest` already bounds
    # the list at min_length=1, so the ValueError is unreachable from the wire.
    pair_code = store.mint(body.label, grants)

    state: AuthState = request.app.state.auth
    state.limiter.record_success(_PAIR_BUDGET_KEY)

    endpoints = _endpoints(request)
    # The code goes in the fragment, never the path or the query: a fragment
    # is stripped by the browser before the request leaves it, so no server,
    # proxy or tunnel between the phone and this daemon ever sees it.
    svg = qr_svg(f"{endpoints[0]}/pair#{pair_code.code}")

    # `expires_at` on the PairCode is a `time.monotonic()` reading, which
    # means nothing to a client. Converting the remaining seconds rather than
    # recomputing `now + CODE_TTL_SECONDS` keeps this from drifting if the
    # store's TTL ever changes.
    remaining = max(0.0, pair_code.expires_at - time.monotonic())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=remaining)

    # Label, never the code, and never the grants either -- one line saying a
    # window opened is worth having; the contents of the window are not.
    logger.info(f"[API] pair code minted for {body.label!r}")

    return Envelope(data=PairCodePayload(
        code=pair_code.code,
        expires_at=expires_at.isoformat(),
        endpoints=endpoints,
        qr_svg=svg,
    ))


# ─── redeeming: the only unauthenticated write ───────────────────────────
@router.post("/pair", status_code=status.HTTP_204_NO_CONTENT)
async def pair_device(body: PairRequest, request: Request) -> Response:
    """Redeem a code for a device credential, delivered as the cookie.

    No auth dependency, by definition: this is how a device gets a credential
    in the first place. What stands in for one is the code -- ~40 bits, live
    for at most 180 seconds, single-use, at most one outstanding, and mintable
    only from this machine.

    **204 with no body, and the token only in `Set-Cookie`.** Returning the
    token in JSON as well would put a working credential into a response body
    that a client could log, cache, or hand to a template; the cookie is
    `httpOnly`, so the page that just paired cannot read it either, and that
    is the whole point of moving off `localStorage` in Task 5. There is
    nothing else a caller needs back -- it learns what it may do by calling
    `GET /v1/session` with the credential it now holds.

    Every refusal is the same `_UNAUTHORIZED` the rest of the API uses. The
    one status that differs is the 429, and it has to: a caller that is being
    throttled must be able to tell that retrying immediately is pointless,
    and it learns nothing about any code by being told so.
    """
    policy = policy_for_scope(request.scope, request.app.state.listener_policies)
    if policy is None:
        # Nobody declared what this socket is. Same answer `authenticate()`
        # gives, for the same reason: a listener added later and forgotten in
        # the registry must not inherit loopback's rights by silence.
        raise _UNAUTHORIZED

    # Before the budget, and unmetered -- the same ordering and the same
    # reasoning as `authenticate()`. It costs one header lookup and a set
    # membership test, it never touches the vault or the store, and metering
    # it would let one malicious page in one visitor's browser spend the
    # budget belonging to everybody behind a tunnel.
    #
    # The `Origin` half only. This route reads no credential -- its authority
    # is the code in the body -- so there is no ambient authority for the CSRF
    # header to defend, and demanding it would refuse a phone re-pairing with
    # a stale cookie still in its jar, which is the one flow that has to work.
    # See `refuse_unknown_origin` for the full argument.
    refuse_unknown_origin(request, policy)

    state: AuthState = request.app.state.auth
    store = _store(request)

    if not state.limiter.check(_PAIR_BUDGET_KEY,
                               max_per_window=_PAIR_MAX_PER_WINDOW,
                               window_seconds=_PAIR_WINDOW_SECONDS):
        # The budget is spent, so the outstanding code burns. Sustained
        # guessing must not be merely slowed against a code that is still in
        # the air; an attacker is made to beat a *new* 180-second window that
        # only the laptop can open. Idempotent, so it does not matter that
        # every further refused attempt calls it again.
        store.burn()
        logger.warning("[API] pair attempts exhausted the budget; the live "
                       "pair code was destroyed. Mint a new one to pair.")
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="too many requests")

    pair_code = store.consume(body.code)
    if pair_code is None:
        # Wrong, expired, already used, or malformed -- `consume()` returns
        # `None` for all four and this route must not tell them apart.
        state.limiter.record_failure(_PAIR_BUDGET_KEY)
        logger.info("[API] a pair attempt was refused")
        raise _UNAUTHORIZED

    # The label comes from the code, and `PairRequest` carries no label field
    # at all for it to come from anywhere else. It is what the person at the
    # laptop typed while choosing the grants, and it is the text the revoke
    # list is read by -- so a device that could name itself could name itself
    # `laptop` and make that list unreliable at the exact moment somebody is
    # deciding which row to cut off. A `label` key in the body is ignored
    # rather than rejected (`PairRequest` does not forbid extras) so that a
    # client which tries still ends up with the name the laptop authorised,
    # instead of a 422 it could route around by dropping the key.
    token = request.app.state.auth.vault.issue(pair_code.label, pair_code.grants)
    state.limiter.record_success(_PAIR_BUDGET_KEY)
    logger.info(f"[API] paired a new device: {pair_code.label!r}")

    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.set_cookie(COOKIE_NAME, token, **cookie_kwargs(policy))
    return response
