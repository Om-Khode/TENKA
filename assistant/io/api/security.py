# assistant/io/api/security.py
"""Authentication, capability checks, rate limiting, audit.

Authentication is a router-level dependency rather than a per-route decorator,
so a new route is authenticated by construction. Unknown, malformed and revoked
tokens produce one identical response after the same work.

The credential is a cookie. Milestone 6 serves Studio from this daemon and puts
it on a public URL, which changes what a stored token is exposed to: anything
readable by JavaScript (a `localStorage` entry, a token pasted into a URL) is
one XSS in the served UI away from being exfiltrated, and a token in a query
string is written verbatim into every intermediary's access log. An `httpOnly`
cookie is readable by neither. What that buys costs something back, though --
a cookie is *ambient* authority, attached by the browser to any request any
page makes -- so the two gates that ambient authority requires live here too:
a CSRF header on cookie-authenticated writes, and an `Origin` check.

Layering: io/api — core + config only.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

from fastapi import Depends, HTTPException, Request, status
from starlette.requests import HTTPConnection

from .policy import ListenerPolicy, effective, policy_for_port
from .vault import Capability, Device, TokenVault

logger = logging.getLogger(__name__)

# The cookie name is not a secret and does not need to be unguessable -- it
# names the slot, not the credential in it.
COOKIE_NAME = "tenka_device"

# A header a cross-site form post or an <img> cannot set. A browser will only
# attach a custom header to a request its own JavaScript built, and doing that
# cross-origin costs a CORS preflight this daemon never approves -- so a
# non-empty value here proves the request came from a page allowed to talk to
# us, whatever that value happens to say. The value is never *interpreted*;
# it is required only to be present and non-empty, since a header a client
# had to decide to send is the whole signal and an empty one is more likely a
# proxy artefact than a deliberate act.
CSRF_HEADER = "X-TENKA-Request"

# Methods that change something. A GET is not required to carry the CSRF
# header: it is expected to be safe, and requiring it would break the one
# request a browser makes without any script at all (a navigation).
_AMBIENT_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Names that can only mean this machine. `Host` is attacker-controlled, so
# this set is used to *reject* (see `host_is_allowed`), never to decide what a
# request is allowed to do.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="unauthorized",
    headers={"WWW-Authenticate": "Bearer"},
)

# Unlike `_UNAUTHORIZED`, these two may say which one tripped -- and the reason
# is NOT that a credential has already verified by the time they are raised. It
# is the opposite: `_refuse_cross_site()` runs *before* `credential_from()` and
# `verify()` (see `authenticate()`), and neither of its checks ever consults the
# vault. "Did this request come from somewhere allowed to make it?" is
# answerable without knowing whose request it is, so the answer cannot carry
# anything about whose it was. A caller learns exactly two things, both of which
# it already had: which of this daemon's own front doors are allowed, and that
# it is carrying a cookie -- which its own browser sent.
#
# If you are here because you are moving `_refuse_cross_site()` back down below
# `verify()`: don't. That ordering is what these two constants used to
# document, and it made the pair into a credential oracle -- 403 to a
# cross-site request holding a valid cookie, 401 to the identical request
# without one, which is a "does this browser have a working session for that
# daemon?" probe any page could run. `test_a_cross_site_refusal_does_not_reveal_
# whether_the_cookie_is_valid` pins the fix; the event socket's handshake in
# app.py checks Origin before reading the cookie for the same reason.
_CROSS_SITE = HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="origin not allowed")
_CSRF_MISSING = HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                              detail="missing request header")

_WINDOW_SECONDS = 60.0
_MAX_PER_WINDOW = 120
_MAX_FAILURES = 10
_LOCKOUT_BASE_SECONDS = 2.0
_MAX_LOCKOUT_SECONDS = 300.0


@dataclass
class RateLimiter:
    """Per-key sliding window plus exponential lockout on auth failures."""

    hits: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    locked_until: dict[str, float] = field(default_factory=dict)

    def check(self, key: str, now: float | None = None, *,
              max_per_window: int = _MAX_PER_WINDOW,
              window_seconds: float = _WINDOW_SECONDS) -> bool:
        now = time.monotonic() if now is None else now
        if now < self.locked_until.get(key, 0.0):
            return False
        window = self.hits[key]
        while window and now - window[0] > window_seconds:
            window.popleft()
        if len(window) >= max_per_window:
            return False
        window.append(now)
        return True

    def record_failure(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.failures[key] += 1
        if self.failures[key] >= _MAX_FAILURES:
            backoff = min(
                _LOCKOUT_BASE_SECONDS * (2 ** (self.failures[key] - _MAX_FAILURES)),
                _MAX_LOCKOUT_SECONDS,
            )
            self.locked_until[key] = now + backoff

    def record_success(self, key: str) -> None:
        self.failures.pop(key, None)
        self.locked_until.pop(key, None)


@dataclass
class AuditEntry:
    at: str
    device_id: str
    method: str
    path: str
    outcome: str


class AuditLog:
    """Append-only, bounded, in-memory. Surfaced read-only in settings."""

    def __init__(self, capacity: int = 2_000) -> None:
        self._entries: deque[AuditEntry] = deque(maxlen=capacity)

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    def entries(self) -> list[AuditEntry]:
        return list(self._entries)


@dataclass
class AuthState:
    vault: TokenVault
    limiter: RateLimiter = field(default_factory=RateLimiter)
    audit: AuditLog = field(default_factory=AuditLog)


# ─── the credential ──────────────────────────────────────────────────────
def cookie_kwargs(policy: ListenerPolicy) -> dict:
    """Flags for `Response.set_cookie`, chosen by the listener.

    Never a `domain` key, on any policy, and a test pins that for all four.
    A cookie without `Domain` is *host-only*: the browser sends it back to
    exactly the host that set it and to nothing else. With `Domain`, it is
    sent to every subdomain of whatever was named -- and the parents this
    daemon publishes under (`*.ts.net`, `*.trycloudflare.com`) are shared:
    the neighbouring hosts under them are other people's machines. A
    `Domain` attribute there would hand this credential, which reaches
    `code_executor`, to strangers.

    `samesite=strict` because the UI is served by this same daemon from
    Milestone 6 on, so there is no legitimate cross-site navigation into an
    authenticated view to preserve. `secure` follows the transport: required
    everywhere TLS exists, impossible to require on plain-http loopback.
    """
    return {
        "httponly": True,       # an XSS in the served UI cannot read it
        "samesite": "strict",
        "secure": policy.secure_cookie,
        "path": "/",
    }


def cookie_credential(connection: HTTPConnection) -> str:
    """The cookie's value, or `""`. Separate from `credential_from` because
    *which channel* the credential arrived on decides whether CSRF applies."""
    return (connection.cookies.get(COOKIE_NAME) or "").strip()


def credential_from(connection: HTTPConnection, policy: ListenerPolicy) -> str | None:
    """The token this connection presented, or `None`.

    The cookie is read on every listener; `Authorization: Bearer` only where
    the policy allows it, which today is loopback alone. A header survives a
    copy-paste into a script and a shell history in a way a cookie does not,
    which is tolerable on a socket a caller cannot reach without already
    being on this machine and is not tolerable anywhere a network can see.

    The cookie wins when both are present, deliberately: the alternative --
    falling back to the header when the cookie fails to verify -- would make
    the answer to "was this request cookie-authenticated?" depend on whether
    the cookie happened to be valid, and that answer is what decides whether
    CSRF is enforced. One channel per request, chosen before verification.

    Typed on `HTTPConnection`, not `Request`, because the event socket needs
    exactly this logic and `WebSocket` is the other subclass.
    """
    cookie = cookie_credential(connection)
    if cookie:
        return cookie
    if not policy.allow_bearer:
        return None
    scheme, _, value = connection.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


# ─── which listener did this arrive on ───────────────────────────────────
def accepting_port(scope) -> int | None:
    """The local port the connection was accepted on, from the ASGI scope's
    own server address -- the one piece of addressing a client cannot choose.

    Not the client address (`cloudflared` and `tailscale funnel` both connect
    from 127.0.0.1, so a tunnel is indistinguishable from a local caller by
    peer address) and not `Host` (attacker-controlled input picking its own
    permissions). See `policy.py` for the full argument.
    """
    server = scope.get("server")
    if not server or len(server) < 2:
        return None
    port = server[1]
    return port if isinstance(port, int) else None


def policy_for_scope(scope, registry: dict[int, str]) -> ListenerPolicy | None:
    """The listener policy for this connection, or `None` -- which denies."""
    port = accepting_port(scope)
    if port is None:
        return None
    return policy_for_port(port, registry)


# ─── DNS rebinding: the Host allow-list ──────────────────────────────────
def hostname_of(value: str) -> str:
    """The host part of a `Host` header or an origin authority, lowercased."""
    value = value.strip().lower()
    if value.startswith("["):                 # [::1]:8787
        end = value.find("]")
        return value[:end + 1] if end != -1 else value
    return value.split(":", 1)[0]


def host_is_allowed(host_header: str | None, published) -> bool:
    """A rejection gate, never a selection one.

    A page on `evil.example` can re-resolve its own hostname to 127.0.0.1 and
    then speak to this daemon as same-origin -- the browser's origin checks
    all pass, because as far as it knows nothing changed. What does not change
    is the `Host` header it keeps sending: still `evil.example`, because that
    is the name in the address bar. Refusing every name that is not one of
    ours is the standard defence, and it works precisely because the attacker
    cannot alter that header from a page.

    The port in `Host` is ignored. It carries no security meaning here (the
    real port is the one the socket was accepted on, and policy already comes
    from there), while insisting on it would reject every tunnel: a transport
    forwards with the *public* authority, whose port is not this daemon's.
    """
    if not host_header:
        # HTTP/1.1 requires a Host header. Something that omits it is either
        # broken or hand-built, and neither needs to be served.
        return False
    name = hostname_of(host_header)
    if not name:
        return False
    if name in _LOOPBACK_HOSTS:
        return True
    return name in {str(h).strip().lower() for h in published}


# ─── the active endpoint set: what may talk to us ────────────────────────
def endpoint_origins(app_state, port: int | None,
                     policy: ListenerPolicy) -> frozenset[str]:
    """Every origin this daemon considers to be one of its own front doors.

    Built from the port the connection was accepted on (so a second listener
    cannot lend its origins to the first), plus whatever hostnames a running
    transport has published, plus -- on a `local` listener only -- the
    configured development origins. Studio runs on its own dev server during
    development and is served same-origin in production; letting the dev list
    through on a tunnelled listener would mean a public URL trusting an origin
    that only ever existed on someone's laptop.
    """
    origins: set[str] = set()
    if port is not None:
        origins.add(f"http://127.0.0.1:{port}")
        origins.add(f"http://localhost:{port}")
    for host in getattr(app_state, "published_hosts", ()) or ():
        # A published transport hostname is always reached over TLS: both
        # Tailscale and Cloudflare terminate https for it.
        origins.add(f"https://{str(host).strip().lower()}")
    if policy.name == "local":
        for origin in getattr(app_state, "cors_origins", ()) or ():
            origins.add(str(origin).strip().lower().rstrip("/"))
    return frozenset(origins)


def origin_is_known(origin: str, app_state, port: int | None,
                    policy: ListenerPolicy) -> bool:
    """Exact match on scheme, host *and* port, unlike the `Host` gate.

    A browser computes `Origin` itself and a page cannot forge it, so it is
    the one field precise enough to be worth matching precisely -- and the
    precision buys something real: a cookie is host-scoped and ignores ports
    entirely, so another web app listening on a different port of this same
    machine is same-host as far as the cookie jar is concerned. Port-exact
    origin matching is what stops that app's pages from driving this daemon.
    """
    return origin.strip().lower().rstrip("/") in endpoint_origins(app_state, port, policy)


def device_key(device: Device) -> str:
    """The limiter key a verified device spends, never a source address.

    Shared by `authenticate()`, `throttle()`, and the event socket in
    app.py -- one spelling of "this key belongs to this device" rather than
    three copies that could drift apart.
    """
    return f"device:{device.device_id}"


async def authenticate(request: Request) -> Device:
    """Verify first, then spend budget on the right key.

    Two separate budgets, not one shared by everybody behind the same
    address. A request that verifies to a real device is metered by that
    device's own id, so one caller sharing a NAT/CGNAT address -- common
    for the project's India-based users, and the default once Milestone 6
    puts every request behind one tunnel -- can never exhaust a different
    device's throughput. A request that does not verify (no token, a
    malformed header, an unknown, revoked, or wrong token) has no device
    identity to meter, so it is charged against the source address instead.
    Verifying before checking either budget also means a valid token is
    never refused a 429 it never earned just because other traffic from the
    same address already burned that address's budget.

    Failure accounting is asymmetric on purpose, decided here after Task 10
    reverted the same change when it landed as an untested side effect: a
    *wrong* token is a credential guess and still spends the lockout
    budget (`record_failure`) exactly as before, but a request presenting
    no token at all has nothing to guess with and never does. An anonymous
    flood that reaches the vault is still bounded -- the sliding window above
    caps it at roughly `_MAX_PER_WINDOW` requests per `_WINDOW_SECONDS`,
    sustained indefinitely -- it simply never escalates into the exponential,
    multi-minute lockout that a wrong-token guesser earns.

    "Reaches the vault" is the honest scope of that bound, and the qualifier
    is load-bearing: a request refused by `_refuse_cross_site()` below is
    turned away before any budget is consulted, so a flood of cross-site
    requests is not metered here at all. That is deliberate, twice over.
    Such a request costs two header lookups and a set membership test -- no
    HMAC, no `devices.json` read, strictly less than the 421 the `Host` gate
    in app.py already answers unmetered and outside this function entirely --
    so there is no work worth rationing. And metering it would be actively
    harmful once a tunnel exists: `cloudflared` and `tailscale funnel` both
    connect from 127.0.0.1, so `source` collapses to a single shared key for
    every remote caller, and one malicious page running in one visitor's
    browser could then spend the anonymous budget belonging to everybody
    behind the tunnel. A cheap refusal that cannot be turned into a denial of
    service against honest callers is the better trade.

    The `Device` this returns is the device *as this listener sees it*: its
    grants have already been intersected with the listener's ceiling, so every
    caller downstream -- `require()`, and `routes/commands.py`, which checks a
    grant of its own choosing against `device.grants` without going through
    `require()` at all -- inherits the narrowing for free. Doing it here rather
    than inside `require()` is the difference between a ceiling that applies
    to every route and one that applies to the routes that remembered to ask.
    """
    state: AuthState = request.app.state.auth
    source = request.client.host if request.client else "unknown"

    policy = policy_for_scope(request.scope, request.app.state.listener_policies)
    if policy is None:
        # Nobody declared what this socket is, so it carries nothing. Failing
        # closed here is the whole point of keying policy on the port: a
        # listener added in a later milestone and forgotten in the registry
        # serves 401s, rather than quietly inheriting loopback's rights.
        if not state.limiter.check(source):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="too many requests")
        raise _UNAUTHORIZED
    request.state.policy = policy

    # Before `verify()`, deliberately, and matching the order the event socket
    # already used. Run afterwards, these two checks answered 403 to a
    # cross-site request holding a valid cookie and 401 to the same request
    # without one -- a "does this browser hold a working credential for this
    # daemon?" oracle that any page could query. Neither check needs to know
    # whether the credential is good, so neither should be able to reveal it.
    _refuse_cross_site(request, policy)

    token = credential_from(request, policy) or ""
    device = state.vault.verify(token)

    if device is not None:
        key = device_key(device)
        if not state.limiter.check(key):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="too many requests")
        state.limiter.record_success(key)
        grants = effective(device.grants, policy)
        if not grants:
            # This transport carries nothing this device holds. Refused as an
            # authentication failure, not a 403: a device that authenticates
            # with an empty grant set can still reach any route gated on
            # `authenticate` alone, which turns the 404-vs-403 split in
            # `run_command` into an oracle for which command ids exist.
            # `TokenVault.issue()` refuses an empty grant set at the source
            # for exactly this reason; the ceiling is the other way to arrive
            # at one, and it is closed the same way.
            raise _UNAUTHORIZED
        device = replace(device, grants=grants)
        request.state.device = device
        return device

    if not state.limiter.check(source):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="too many requests")
    if token:
        state.limiter.record_failure(source)
    raise _UNAUTHORIZED


def _refuse_cross_site(request: Request, policy: ListenerPolicy) -> None:
    """The price of an ambient credential, paid before the credential is read.

    A cookie is attached by the browser to *any* request *any* page makes to
    this host. Two checks make that safe, and they cover different gaps:

    - `Origin`, when the browser sent one. Checked on every method, not just
      writes, because a cross-site read of `/v1/files` or `/v1/memory` is a
      disclosure even though it changes nothing. Only checked when present:
      a browser omits `Origin` on a same-origin GET and on a plain
      navigation, so demanding it would break the UI this daemon serves.
    - The CSRF header, on writes, when the credential was the cookie. This is
      what covers the case `Origin` cannot: a cross-site form post or an
      `<img>` carries no `Origin` a browser is obliged to send, and needs no
      script, so there is nothing for the check above to catch.

    A *bearer*-authenticated write is exempt. CSRF exists because the browser
    supplies the credential without the page asking; an `Authorization` header
    is never attached automatically, so a cross-site page cannot produce one
    at all -- there is no forgeable request to defend against. Bearer is
    loopback-only besides. Demanding the header there would only break `curl`.

    Neither check consults the vault: "did this request come from somewhere
    allowed to make it?" is answerable without knowing whose request it is,
    which is what lets both run ahead of verification and keeps the refusal
    from doubling as a credential oracle.
    """
    port = accepting_port(request.scope)
    origin = request.headers.get("Origin")
    if origin and origin.strip():
        if not origin_is_known(origin, request.app.state, port, policy):
            raise _CROSS_SITE
    if request.method.upper() in _AMBIENT_METHODS and cookie_credential(request):
        if not request.headers.get(CSRF_HEADER):
            raise _CSRF_MISSING


def require(capability: Capability):
    """Dependency factory: authenticate, then insist on one grant.

    `device.grants` here is already `effective(issued, listener ceiling)` --
    `authenticate()` narrows before it returns -- so this one comparison is
    both "was the device granted this?" and "does this transport carry it?".
    """

    async def _dependency(device: Device = Depends(authenticate)) -> Device:
        if capability not in device.grants:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="capability not granted")
        return device

    return _dependency


def throttle(capability: Capability, scope: str, *,
             max_per_window: int, window_seconds: float = _WINDOW_SECONDS):
    """Dependency factory: `require(capability)`, then a tighter per-route
    budget stacked on top of the shared limiter.

    Generic by construction -- `scope` is any short label the caller picks,
    not an app name -- for a route whose cost is not the same as an ordinary
    read. A device that can legitimately call a cheap route 120 times a
    minute should not, by the same token, get to trigger 120 real cloud
    uploads a minute; the shared budget bounds total throughput, this bounds
    one expensive route's share of it. Keyed by device (never by source),
    so it inherits `authenticate()`'s fairness fix rather than reopening the
    shared-address problem for exactly the routes that most need throttling.
    """

    async def _dependency(request: Request,
                          device: Device = Depends(require(capability))) -> Device:
        state: AuthState = request.app.state.auth
        key = f"{scope}:{device_key(device)}"
        if not state.limiter.check(key, max_per_window=max_per_window,
                                    window_seconds=window_seconds):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="too many requests")
        return device

    return _dependency
