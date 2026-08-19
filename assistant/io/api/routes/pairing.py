# assistant/io/api/routes/pairing.py
"""Minting a pair code, and redeeming one.

`POST /v1/pair` is the only unauthenticated write in this API. From Milestone
6b it is reachable from the open internet, and it sits beside a `POST /v1/chat`
that reaches `code_executor` -- so what this module refuses matters more than
what it serves. These properties carry that weight, and each one has a test:

- **Enrollment is loopback-only.** `POST /v1/pair/code` is gated on
  `require_admin(SYSTEM_CONTROL)`: a remote listener is never `policy.admin`,
  so a compromised phone cannot mint itself a second, wider credential at all.
- **A minted code can never carry more than the minting device already holds.**
  `require_admin` proves SYSTEM_CONTROL and a loopback listener, and nothing
  else -- it says nothing about whether this device also holds, say, RECALL or
  FILES. A device issued only SYSTEM_CONTROL used to be able to mint a code
  carrying every other capability there is, a second credential wider than the
  one it was issued, through the one route that exists specifically to prevent
  that escalation. Requested grants are intersected with the minting device's
  own before the code is minted, never unioned with them.
- **Grants that survive to `issue()` are also capped by what the redeeming
  listener could EVER carry.** The code carries what the laptop authorised;
  `pair_device` below narrows that further to `effective(pair_code.grants,
  policy, raised=policy.raisable)` before it ever reaches the vault, so a
  code redeemed over `tailnet` or `funnel` can never mint a device wider than
  `ceiling | raisable` for that transport -- narrowing to the bare `ceiling`
  instead used to discard a raisable capability from the device's own issued
  set permanently, which is a defect, not the boundary: a raise can already
  never manufacture a capability the device side lacks (`policy.py`'s
  invariant), so issue-time narrowing has nothing further to protect by
  cutting `raisable` capabilities off before a raise ever gets a chance to
  reach them.
- **Redemption is refused outright on `quick`.** Cloudflare terminates TLS and
  reads the plaintext of the code and the resulting `Set-Cookie` alike, so
  narrowing grants there is not enough -- a third party would still see a
  working, if watch-only, credential in the clear. Refused before the code is
  even consulted, so a wrong code and the transport's one genuinely live code
  read identically. See `pair_device`'s own docstring for the full argument.
- **Grants ride on the code.** They are chosen on the laptop before the QR is
  drawn and stored on the `PairCode`; the redeeming request never supplies
  them. That is what turns the checkbox row into a boundary rather than a
  suggestion.
- **Wrong, expired and already-used codes are one response.** `consume()`
  already collapses all three to `None`; this module collapses them to one
  `UNAUTHORIZED` after the same work, exactly as `verify()` does for
  unknown-vs-revoked tokens.
- **No log line ever carries a code.** Not at DEBUG, not in an audit entry.
  The code appears in exactly one place, the response body that the laptop
  renders, and in the QR's URL *fragment* -- which a browser never sends to a
  server, so it cannot land in an intermediary's access log either. That
  fragment is what the laptop's own `_endpoints()`/`transport` handling builds
  from -- loopback origin on `local`, or a named transport's published
  `https://` host -- never the path or the query, on either shape.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from ..events import notify_invalidate
from ..pairing import PairCodeStore
from ..payloads import PairCodePayload
from ..policy import POLICIES, ListenerPolicy, effective, policy_for_port
from ..qr import qr_svg
from ..schemas import Envelope, PairCodeRequest, PairRequest
from ..security import (
    UNAUTHORIZED,
    AuthState,
    accepting_port,
    cookie_kwargs,
    cookie_name_for,
    policy_for_scope,
    refuse_unknown_origin,
    require_admin,
)
from ..vault import Capability, Device, VaultUnavailableError

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
# The first half of that -- "every remote caller already shares one source
# key" -- is only true because `server.py` now passes `proxy_headers=False`
# to uvicorn. Under uvicorn's own default it was false:
# `ProxyHeadersMiddleware` rewrote `scope["client"]` from `X-Forwarded-For`
# for precisely the loopback peers this comment names, so
# `request.client.host` *was* the attacker-supplied header the next sentence
# rejects, and this route's conclusion was right for a reason that did not
# hold. Both halves hold now. The fixed key stays correct either way; what
# breaks first if anybody re-enables proxy headers is `authenticate()`'s
# source keying -- see its docstring.
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
    """Every base URL a phone could be told to reach THIS listener at.

    Listener-scoped, from the port this connection was *accepted* on -- the
    one piece of addressing a client cannot choose. On the loopback listener,
    the loopback origin, exactly as in 6a. On a transport listener, that
    listener's published origins (`PublishedHosts.origins_for`), each already
    carrying that transport's own public port (`8443` on `tailnet` -- the
    live-test defect this fixes: a phone off-LAN cannot reach `127.0.0.1`,
    and a bare `https://{host}` with no port is just as unreachable, since
    it resolves to 443 where nothing is listening). `hosts_for` -- bare
    hostname, no port -- stays the `Host` gate's own read (KI-17's
    load-bearing layer); `origins_for` is the separate, port-carrying read
    this function and `endpoint_origins` (`security.py`) share, so the QR
    this function feeds and the CORS/CSRF origin set that must accept a
    request back from it agree on the identical string by construction
    rather than by two call sites happening to compute the same port.

    6a's "deliberately not read here yet" note about `app.state.published_hosts`
    is resolved -- but not by this function reaching past its own listener.
    The resolution is `mint_pair_code`'s `transport` handling below, the one
    deliberate cross-listener read this milestone introduces: minting always
    happens on `local` (`require_admin`), so this function alone would only
    ever answer with the loopback origin, and a named transport's own hosts
    have to be read on purpose, explicitly, rather than by this function
    guessing which listener the operator meant. Advertising an endpoint is
    not trusting an origin.
    """
    port = accepting_port(request.scope)
    if port is None:
        # Unreachable: `authenticate()` resolves policy from this same port
        # and refuses when it cannot. Guarded anyway rather than formatting
        # `None` into a URL a QR would then encode.
        raise UNAUTHORIZED
    policy = policy_for_port(port, request.app.state.listener_policies)
    if policy is not None and policy.name == "local":
        return [f"http://127.0.0.1:{port}"]
    published = request.app.state.published_hosts
    return sorted(published.origins_for(port))


def pairing_denied_by_transport(policy: ListenerPolicy) -> bool:
    """Whether `pair_device` refuses redemption outright on this listener,
    before the code is even consulted -- spec §5.5.

    `quick` alone, today: Cloudflare terminates TLS and would read both the
    code and the resulting `Set-Cookie` device token in the clear, so
    narrowing grants there is not enough. The one place this fact lives, so
    `GET /v1/listener`'s `can_pair` field can answer "would `pair_device`
    refuse me?" by calling this function rather than re-spelling
    `policy.name == "quick"` a second time somewhere else -- a route reading
    a copy of the condition is exactly how the two drift apart.
    """
    return policy.name == "quick"


# ─── minting: loopback only ──────────────────────────────────────────────
@router.post("/pair/code")
async def mint_pair_code(
    body: PairCodeRequest, request: Request,
    device: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[PairCodePayload]:
    """Put a code, and a QR for it, in front of the person at the keyboard.

    Minting also clears the pair route's failure lockout. That is what makes
    the accepted denial of service above tolerable rather than permanent: an
    attacker grinding at a code can lock the pairing window shut, and the one
    person who can reopen it is somebody standing at this machine holding
    SYSTEM_CONTROL -- precisely the person being denied. It is not a way in
    for the attacker, because reaching this line at all already required
    everything the attacker does not have.

    **Requested grants are capped at what this device already holds.**
    `require_admin` proves this device holds SYSTEM_CONTROL and is on a
    loopback listener -- it proves nothing about whether it also holds any
    *other* capability being asked for. Without the intersection below, a
    device paired with only SYSTEM_CONTROL could mint a code carrying RECALL,
    FILES and CHAT_SEND -- none of which it was ever issued -- and redeem that
    code itself for a second credential wider than the one it holds.
    `Capability`'s own docstring is explicit that grants are "granted per
    device, never implied": SYSTEM_CONTROL does not subsume anything else, so
    neither does minting with it.
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

    # Intersected, never unioned, with the minting device's own grants -- the
    # same shape as `effective()` narrowing a listener's ceiling, applied here
    # to a *device's* ceiling instead of a transport's. `PairCodeRequest`
    # bounds the requested list at min_length=1, but the intersection can
    # still land empty if none of what was asked for is actually held, so
    # that case gets its own 422 rather than reaching `store.mint()`'s
    # (unreachable-from-the-wire, until now) empty-grant refusal.
    grants = grants & device.grants
    if not grants:
        raise HTTPException(
            status_code=422,
            detail="none of the requested capabilities are held by this device",
        )

    # `transport` is resolved before the code is minted, never after:
    # `store.mint()` replaces whatever code is already live, and a request
    # naming a transport that turns out not to be running must not destroy an
    # operator's still-live code as a side effect of failing.
    #
    # `"local"` -- the default, and 6a's only shape -- reads `_endpoints()`
    # above, which is scoped to the listener this request itself arrived on.
    # Minting is always accepted on `local` (`require_admin`), so that is
    # always the loopback origin. Naming a *different* transport is this
    # milestone's one deliberate cross-listener read: `_endpoints()` would
    # only ever answer loopback here, which is useless to a phone off-LAN,
    # so the operator names the transport explicitly and this route reads
    # THAT transport's own published host instead. Advertising an endpoint is
    # not trusting an origin.
    if body.transport == "local":
        endpoints = _endpoints(request)
    else:
        if body.transport not in POLICIES:
            # Same silence about the submitted string as the capability
            # check above: an unknown transport name is a client mistake,
            # never echoed back into a 422 body.
            raise HTTPException(status_code=422, detail="unknown transport")
        # `getattr`, the same defensive shape `raise_device_ceiling` in
        # `routes/devices.py` uses for the identical reason: a sibling task
        # wires the transport manager, and an app built without one -- or one
        # wired but with nothing running under this name -- must fail closed
        # rather than fall back to loopback. `running()` is already filtered
        # on `is_serving`, so a session that started but never announced a
        # hostname is absent from it too; "not running" and "no published
        # host yet" are one refusal here, not two, because a scannable code
        # with no destination is worse than no code either way.
        manager = getattr(request.app.state, "transports", None)
        running = manager.running() if manager is not None else {}
        session = running.get(body.transport)
        if session is None or session.url is None:
            raise HTTPException(
                status_code=409,
                detail="that transport is not running, or has not yet "
                       "published a hostname",
            )
        endpoints = [session.url]

    store = _store(request)
    pair_code = store.mint(body.label, grants)

    state: AuthState = request.app.state.auth
    state.limiter.record_success(_PAIR_BUDGET_KEY)

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

    **Grants are narrowed at issue time to what this listener could EVER
    carry -- `ceiling | raisable`, not `ceiling` alone.**
    `effective(pair_code.grants, policy, raised=policy.raisable)`, never the
    code's own unnarrowed set and, as importantly, never `effective(grants,
    policy)`'s everyday two-argument form either. The difference is the fix
    for a live-test defect: narrowing to the bare ceiling at issue time
    discarded a raisable capability from the *device's own issued set*,
    permanently, the moment a code was redeemed on that transport -- and a
    raise can only ever widen the *transport* side of `effective()`'s
    intersection (`policy.py`'s own invariant), never manufacture a
    capability the device side does not already hold. So a code minted with
    EXECUTE and SYSTEM_CONTROL, redeemed over `tailnet` (`raisable =
    {EXECUTE, SYSTEM_CONTROL}`), used to mint a device that could never
    reach either capability again on that transport, by any raise, ever --
    confirmed live: ticking both boxes and pairing over tailnet stored
    neither. `tailnet`'s whole `raisable` set was unreachable configuration.

    Reusing `effective()` with `raised=policy.raisable` rather than adding a
    new function is deliberate -- the arithmetic
    `grants & (ceiling | (raisable & raisable))` collapses to
    `grants & (ceiling | raisable)` by construction, so this is the *same*
    intersection every other call site already trusts, evaluated at the one
    raised value that means "everything this transport could ever carry,"
    not a second narrowing rule to keep in sync with the first.

    **This does not touch per-request narrowing.** `authenticate()` still
    calls `effective()` with whatever the *live* raise store holds for this
    device and listener, not `policy.raisable` -- so a device that paired
    over `tailnet` and now has EXECUTE sitting in the vault is still refused
    EXECUTE on every ordinary request. The only thing this change does is
    make the capability *reachable at all* by a live raise later; it does
    not hand it out for free. That is the property the whole milestone rests
    on, and it is what `funnel` and `quick` prove by their absence: both have
    an empty `raisable`, so `ceiling | raisable` there is just `ceiling` --
    identical to the old behaviour, and neither transport's redemption
    changes shape at all. `funnel` was deliberately left without
    SYSTEM_CONTROL in its own `raisable` for a different reason (a public URL
    with no second credential gating it), which this narrowing respects
    rather than overrides: `funnel`'s ceiling still excludes EXECUTE and
    SYSTEM_CONTROL from what any device paired there ever holds.

    **`quick` is refused outright, spec §5.5, before the code is even
    consulted -- `pairing_denied_by_transport()` just below the policy
    lookup.** Cloudflare terminates TLS and reads the plaintext of
    everything this listener carries, including the code in this body and
    the `Set-Cookie` this route would otherwise write; narrowing to OBSERVE
    alone would still hand a third party a working (if watch-only) device
    credential in the clear, which is a worse trade than refusing to pair
    over that transport at all. Refused before `store.consume()` runs, so a
    wrong code and a right one are indistinguishable on `quick` and no
    attempt -- however it would have turned out -- burns the operator's live
    code.

    Every refusal is the same `UNAUTHORIZED` the rest of the API uses, with
    two exceptions, both load-bearing: 429 when the pairing budget is spent
    (a caller has to be able to tell that retrying immediately is pointless),
    and 503 when the vault itself could not be read or written (a caller has
    to be able to tell that retrying *at all* might still work -- unlike a
    wrong code, this one was never the caller's fault).
    """
    policy = policy_for_scope(request.scope, request.app.state.listener_policies)
    if policy is None:
        # Nobody declared what this socket is. Same answer `authenticate()`
        # gives, for the same reason: a listener added later and forgotten in
        # the registry must not inherit loopback's rights by silence.
        raise UNAUTHORIZED

    if pairing_denied_by_transport(policy):
        # Spec §5.5. Before the budget, before `refuse_unknown_origin`, and
        # before `store.consume()` -- the code is never even looked at, so a
        # wrong code and this transport's one genuinely live code are
        # answered identically and neither attempt costs the budget or burns
        # anything. Cloudflare terminates TLS and reads the plaintext of this
        # whole exchange, code and the `Set-Cookie` this route would
        # otherwise write alike; narrowing to `quick`'s ceiling (OBSERVE
        # alone) still hands a third party a working, if watch-only, device
        # credential in the clear, which is the trade this refuses outright
        # rather than take.
        raise UNAUTHORIZED

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
        raise UNAUTHORIZED

    # Narrowed to what THIS listener could EVER carry -- ceiling union
    # raisable -- see the docstring above. Never the code's own, unnarrowed
    # `pair_code.grants`, and never the bare-ceiling `effective(grants,
    # policy)` either: that form is what made a raisable capability
    # permanently unreachable the moment a code was redeemed.
    grants = effective(pair_code.grants, policy, raised=policy.raisable)
    if not grants:
        # The code was genuinely correct -- `consume()` above already proved
        # that and burned it as single-use -- but this listener cannot carry
        # a single one of the capabilities it was minted with (e.g. a code
        # minted without OBSERVE, redeemed over `quick`, whose ceiling is
        # OBSERVE alone). Answered exactly like a wrong code, and guarded
        # explicitly rather than left to reach `TokenVault.issue()`'s own
        # refusal of an empty grant set: that refusal is a `ValueError`,
        # which `errors.py`'s app-wide mapping turns into 400 "invalid
        # request" -- a status distinct from this route's uniform 401,
        # which would let a caller distinguish "this code was real, single-
        # use, and got this far" from "wrong/expired/reused" by status code
        # alone. That is a valid-code oracle over an unauthenticated route,
        # the same class of leak `consume()` already collapses wrong,
        # expired and reused into one answer to prevent. Not counted as a
        # guessing failure (`record_failure`) -- this caller presented a real
        # code, so charging the pairing budget for a channel limitation would
        # punish correct use of a transport that was always going to end this
        # way for this particular code.
        logger.info("[API] a pair attempt succeeded but this listener could "
                    "carry none of the code's grants")
        raise UNAUTHORIZED

    # The label comes from the code, and `PairRequest` carries no label field
    # at all for it to come from anywhere else. It is what the person at the
    # laptop typed while choosing the grants, and it is the text the revoke
    # list is read by -- so a device that could name itself could name itself
    # `laptop` and make that list unreliable at the exact moment somebody is
    # deciding which row to cut off. A `label` key in the body is ignored
    # rather than rejected (`PairRequest` does not forbid extras) so that a
    # client which tries still ends up with the name the laptop authorised,
    # instead of a 422 it could route around by dropping the key.
    #
    # Off the event loop: `issue()` rewrites `devices.json` and then spawns
    # `icacls` synchronously (tens of milliseconds, measured), and this daemon
    # shares its loop with the assistant -- an inline write stalls her and
    # every other caller, not just this one request. `TokenVault` locks the
    # whole load-append-save sequence itself, so a worker thread is exactly
    # what that lock is for.
    try:
        token = await asyncio.to_thread(
            request.app.state.auth.vault.issue, pair_code.label, grants)
    except VaultUnavailableError as exc:
        # The code is already consumed and gone -- single-use, and
        # `consume()` never puts it back -- so this is not a "retry with the
        # same code" situation. `issue()` refused to write from a
        # devices.json it could not read, or a write that could not land,
        # rather than guess an empty vault and silently destroy every other
        # device, or leave a `PermissionError` to surface as `errors.py`'s
        # 403 "protected path" -- a status that would say "this code is not
        # allowed" when the truth is "the vault is temporarily unavailable"
        # (see `VaultUnavailableError`'s docstring). Whoever is at the laptop
        # mints a fresh code once whatever is holding the file -- a scanner,
        # a backup tool, a second TENKA process -- releases it.
        logger.warning(f"[API] pairing failed: vault unavailable ({exc})")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="try again")
    state.limiter.record_success(_PAIR_BUDGET_KEY)
    logger.info(f"[API] paired a new device: {pair_code.label!r}")
    # After the vault write above has actually landed, never before: the
    # devices list an admin viewer holds open just went stale, and this is
    # the one call that tells it so. See `notify_invalidate` for why this
    # can never turn the pairing that already succeeded into a 500.
    notify_invalidate(getattr(request.app.state, "hub", None), "devices")

    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    # Prefixed on any listener that can set `Secure` -- see the same call in
    # routes/session.py and `HOST_COOKIE_NAME` in security.py. This is the one
    # that matters most: pairing is what a phone does over a tunnel, which is
    # exactly where the shared parent domain has neighbours on it.
    response.set_cookie(cookie_name_for(policy), token, **cookie_kwargs(policy))
    return response
