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

import asyncio
import logging
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request, status
from starlette.requests import HTTPConnection

from ...core.redact import redact_secrets
from .policy import ListenerPolicy, effective, policy_for_port
from .vault import Capability, Device, TokenVault, VaultUnavailableError

logger = logging.getLogger(__name__)

# The cookie name is not a secret and does not need to be unguessable -- it
# names the slot, not the credential in it.
COOKIE_NAME = "tenka_device"

# The same slot, under the one cookie name a browser polices for us.
#
# `__Host-` is not decoration. A browser refuses to *store* a cookie with this
# prefix unless it is `Secure`, has `Path=/`, and carries **no `Domain`
# attribute at all** -- and that last clause is the whole point. Without it,
# any sibling host under a shared parent (`*.ts.net`, `*.trycloudflare.com`:
# the parents 6b publishes under, where the neighbours are other people's
# machines) can set `tenka_device` with `Domain=.ts.net` and have the browser
# send it inward to this daemon. RFC 6265 s5.4 serialises equal-path cookies
# oldest-first, so the attacker's later-planted duplicate is the one a
# last-wins parser adopts.
#
# The server-side duplicate rejection below was the previous repair and it is
# half a fix in both directions. It converts the fixation into a *permanent
# denial of service*: our cookie is host-only, so `set_cookie` cannot delete a
# parent-domain one -- a different `Domain` is a different cookie -- and there
# is no route that could. Every request 401s, and re-pairing re-locks it,
# because the plant is still there. And it does not close the fixation it was
# written for: with no cookie held yet the count is 1, so the planted value is
# simply adopted.
#
# With the prefix, the sibling cannot plant this name at all, and the read
# order below prefers it -- so the daemon's own cookie always wins over
# anything a neighbour writes under the unprefixed name.
#
# It cannot be the only name, because it demands `Secure` and the `local`
# policy serves plain http on loopback (see `cookie_kwargs`). So the name is
# chosen per listener by `cookie_name_for`, and both names are *read*: that
# is what keeps a device paired before this change working until it next
# pairs, and it costs nothing, because a cookie under the prefixed name is
# strictly more trustworthy than one under the plain name and is preferred.
#
# Loopback keeps the unprefixed name and the duplicate rule. It has no shared
# parent to be a sibling of, so what remains there is "another port on
# 127.0.0.1", which needs code already running on this machine.
HOST_COOKIE_NAME = "__Host-tenka_device"

# Preferred first. A `__Host-` cookie can only have been set by this daemon
# over TLS on this exact host; an unprefixed one is whatever reached the jar.
_COOKIE_NAMES = (HOST_COOKIE_NAME, COOKIE_NAME)

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

UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="unauthorized",
    headers={"WWW-Authenticate": "Bearer"},
)

# Unlike `UNAUTHORIZED`, these two may say which one tripped -- and the reason
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

# Device enrollment, the device listing and revocation are refused on any
# listener that is not `policy.admin` -- loopback alone. Deliberately the same
# shape and status as `require()`'s capability refusal: a caller that already
# authenticated learns "this connection may not do that" and not which of the
# two gates said so, so a remote session cannot map out which of its
# capabilities are transport-limited and which are grant-limited.
_NOT_ADMIN = HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                           detail="capability not granted")

_WINDOW_SECONDS = 60.0
_MAX_PER_WINDOW = 120
_MAX_FAILURES = 10
_LOCKOUT_BASE_SECONDS = 2.0
_MAX_LOCKOUT_SECONDS = 300.0

# Eviction. Every distinct key this limiter has ever metered used to keep a
# dict slot for the life of the process -- ~862 bytes measured, never
# reclaimed -- which was defended by the claim that a tunnel collapses every
# remote caller onto one shared source key. That claim was false while
# uvicorn's `proxy_headers` default was in force (see `server.py`, which now
# turns it off), and it would be false again the moment anything upstream
# started keying on a per-caller value. So the growth is bounded here, in the
# collection itself, rather than by an argument about who can reach it: a key
# whose window has fully slid and whose lockout has expired is idle, carries
# no state worth keeping, and is dropped from all three dicts together.
#
# `_PRUNE_ABOVE` keeps the scan off small deployments entirely (one laptop,
# a handful of keys, nothing to reclaim). The sweep itself runs at most once
# per window, so it costs O(keys) per `_WINDOW_SECONDS` rather than per
# request. `_MAX_TRACKED_KEYS` is the hard ceiling for the case the sweep
# cannot help with -- more distinct keys genuinely live inside one window
# than this process is willing to remember.
_PRUNE_ABOVE = 1_000
_MAX_TRACKED_KEYS = 50_000


@dataclass
class RateLimiter:
    """Per-key sliding window plus exponential lockout on auth failures.

    Bounded: see `_prune` for what is dropped and why an idle key is safe to
    forget. Nothing that is currently locked out is ever evicted -- forgetting
    a lockout would hand back the budget it exists to withhold, which is the
    one direction eviction must never move in.
    """

    hits: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    locked_until: dict[str, float] = field(default_factory=dict)
    # -inf, not 0.0: the very first `check()` should be free to sweep, and a
    # test driving `now` from 0.0 must not look like "already swept at 0".
    _pruned_at: float = field(default=float("-inf"), repr=False)

    def check(self, key: str, now: float | None = None, *,
              max_per_window: int = _MAX_PER_WINDOW,
              window_seconds: float = _WINDOW_SECONDS) -> bool:
        now = time.monotonic() if now is None else now
        # Before the key is touched, so a sweep can never drop the entry this
        # very call is about to create.
        self._maybe_prune(now, window_seconds)
        if now < self.locked_until.get(key, 0.0):
            return False
        window = self.hits[key]
        while window and now - window[0] > window_seconds:
            window.popleft()
        if len(window) >= max_per_window:
            return False
        window.append(now)
        return True

    def _maybe_prune(self, now: float, window_seconds: float) -> None:
        if len(self.hits) < _PRUNE_ABOVE:
            return
        # The longest window any caller meters on, never the caller's own:
        # `throttle()` hands this method a route-specific window, and pruning
        # a key against a shorter window than the one it was recorded under
        # would forget state that is still live for somebody else.
        window = max(window_seconds, _WINDOW_SECONDS)
        if now - self._pruned_at <= window:
            return
        self._pruned_at = now
        self._prune(now, window)

    def _prune(self, now: float, window_seconds: float) -> None:
        """Drop every key that is idle: an empty sliding window and no live
        lockout.

        A dropped key's `failures` count goes with it, and that is deliberate
        rather than overlooked. To be dropped, a key must have gone a full
        window with no traffic at all *and* have no lockout outstanding -- so
        the most a guesser recovers by waiting that long is a reset of a
        counter whose only effect is the lockout it has already sat out. The
        sliding window still bounds it at `_MAX_PER_WINDOW` either way, and
        against a 256-bit token neither bound is what is doing the work.
        """
        for key in list(self.hits):
            window = self.hits[key]
            while window and now - window[0] > window_seconds:
                window.popleft()
            if window or now < self.locked_until.get(key, 0.0):
                continue
            del self.hits[key]
            self.failures.pop(key, None)
            self.locked_until.pop(key, None)
        # The other two dicts, swept on their own terms rather than only as a
        # side effect of the loop above. `record_failure` writes to both
        # without going through `hits`, so a key that only ever failed would
        # otherwise be a slot this sweep can never see.
        for key in list(self.failures):
            if key not in self.hits and now >= self.locked_until.get(key, 0.0):
                self.failures.pop(key, None)
                self.locked_until.pop(key, None)
        for key in list(self.locked_until):
            if key not in self.hits and now >= self.locked_until[key]:
                self.locked_until.pop(key, None)
                self.failures.pop(key, None)
        if len(self.hits) <= _MAX_TRACKED_KEYS:
            return
        # More live keys than this process will remember. Shed the least
        # recently active ones first, and never a locked-out one: the point of
        # the ceiling is memory, and a lockout is the one entry whose absence
        # would be a grant rather than a cost.
        evictable = sorted(
            (key for key in self.hits if now >= self.locked_until.get(key, 0.0)),
            key=lambda k: self.hits[k][-1] if self.hits[k] else float("-inf"),
        )
        for key in evictable[:len(self.hits) - _MAX_TRACKED_KEYS]:
            del self.hits[key]
            self.failures.pop(key, None)
            self.locked_until.pop(key, None)

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


# Long enough that no real route is ever truncated -- the deepest path this
# daemon serves is well under a hundred characters, and the record has to stay
# readable to be worth keeping. Short enough that the field is not a place a
# caller can park kilobytes.
_AUDIT_PATH_MAX = 200
_AUDIT_TRUNCATED = "..."

# The device id `audit_and_tag` writes when nothing authenticated. Named here
# rather than spelled `"-"` in two files, because `AuditLog` now routes on it.
ANONYMOUS_DEVICE_ID = "-"

# How many per-device audit rings this process will hold at once.
#
# Not a guess about attackers: a device id only exists because an admin
# listener enrolled it, so the realistic count is the number of things the
# operator paired -- a laptop, a phone, maybe a wall display. 16 is the safety
# net for a vault that grew unattended, and the least recently written ring is
# dropped whole when it is passed. Deliberately above `_MAX_SOCKETS_PER_DEVICE`
# (8) in events.py, which is the other place this codebase guesses at "how many
# things does one household have".
_MAX_AUDIT_DEVICES = 16


def sanitize_audit_path(path: str) -> str:
    """The stored form of a caller-chosen path: bounded, and printable ASCII.

    Every request is audited, including the ones that never authenticate, so
    this field is written by anyone who can reach the port. Unbounded, it was
    a place to park kilobytes; unfiltered, it was a place to put an ANSI escape
    or a NUL that renders in whatever terminal an operator reads the record in.

    Tab, CR and LF are already gone by the time this is called -- `request.url`
    rebuilds the URL and re-parses it through `urllib.parse.urlsplit`, which
    strips them -- and that control is pinned by its own passing test. This is
    the rest of the class, and the backstop if that ever changes.
    """
    cleaned = "".join(c if 0x20 <= ord(c) <= 0x7E else "?" for c in path)
    if len(cleaned) > _AUDIT_PATH_MAX:
        return cleaned[:_AUDIT_PATH_MAX - len(_AUDIT_TRUNCATED)] + _AUDIT_TRUNCATED
    return cleaned


class AuditLog:
    """Append-only, bounded, in-memory. Surfaced read-only in settings.

    **One ring per writer, not one ring per anonymity class.**

    This began as a single `deque(maxlen=2000)`, which made eviction a
    primitive any caller held: a rate-limited request is still audited, so the
    limiter never slows the flush below one entry per request, and ~2,000
    requests flushed every other entry out of the record an operator reads
    after an incident. The first repair split the ring in two, by whether
    anything authenticated, and claimed that "a flood can only erase other
    floods."

    It could not. Anonymity is not the axis a flood runs along. *Any* single
    paired device -- including an OBSERVE-only wall display, the weakest
    credential this design issues -- wrote into the one authenticated ring at
    the 120 requests/minute the limiter permits, so ~2,000 of them flushed
    every other device's records, the `require_admin` pairing and revocation
    entries included. The device that erases the evidence of its own pairing
    is exactly the device an operator is reading the log to find.

    So the axis is the writer. Each device id gets its own ring of `capacity`,
    the anonymous id keeps its own of `anonymous_capacity`, and nobody's
    traffic can reach anybody else's records. What one device can still do is
    lose its *own* oldest entries, which is the irreducible property of a
    bounded log.

    **`capacity` is per device, and that is a changed meaning.** It was the
    total. Sizing follows from what bounds the number of rings: device ids come
    from `devices.json`, which only an admin listener can add to, so the count
    is operator-controlled and realistically a handful. `_MAX_AUDIT_DEVICES` is
    the safety net rather than the expected case, and the ring least recently
    written to is dropped whole when it is exceeded -- an LRU over rings, so
    the device that has said nothing for longest is the one forgotten.

    The defaults put the worst case at `250 * 16 + 500` = 4,500 entries against
    the old 2,500, and the realistic case (three devices) at well under it.
    That is the memory this trade costs.

    `entries()` merges every ring back into one chronological sequence, by an
    internal counter rather than by the `at` string: two entries can share a
    timestamp, and `GET /v1/audit` reverses this list to show newest first, so
    the order has to be exact rather than approximately right.
    """

    def __init__(self, capacity: int = 250,
                 anonymous_capacity: int = 500,
                 max_devices: int = _MAX_AUDIT_DEVICES) -> None:
        self._seq = 0
        self._capacity = max(1, capacity)
        self._max_devices = max(1, max_devices)
        # Insertion-ordered and re-ordered on every write, so the first key is
        # always the least recently used ring.
        self._by_device: OrderedDict[str, deque[tuple[int, AuditEntry]]] = \
            OrderedDict()
        self._anonymous: deque[tuple[int, AuditEntry]] = deque(
            maxlen=anonymous_capacity)

    def record(self, entry: AuditEntry) -> None:
        # Sanitised here, at the store, rather than at each call site: there
        # are three of them (two in the HTTP middleware, one in the event
        # socket) and a fourth added later would otherwise reintroduce the
        # hole silently.
        entry = replace(entry, path=sanitize_audit_path(entry.path))
        self._seq += 1
        if entry.device_id == ANONYMOUS_DEVICE_ID:
            self._anonymous.append((self._seq, entry))
            return
        ring = self._by_device.get(entry.device_id)
        if ring is None:
            if len(self._by_device) >= self._max_devices:
                # Whole ring, not one entry: a device this daemon has not heard
                # from in longer than any other is the least useful history to
                # keep, and dropping a slice of each would put every device
                # back in reach of every other.
                self._by_device.popitem(last=False)
            ring = deque(maxlen=self._capacity)
            self._by_device[entry.device_id] = ring
        else:
            self._by_device.move_to_end(entry.device_id)
        ring.append((self._seq, entry))

    def entries(self) -> list[AuditEntry]:
        merged: list[tuple[int, AuditEntry]] = list(self._anonymous)
        for ring in self._by_device.values():
            merged.extend(ring)
        merged.sort(key=lambda pair: pair[0])
        return [entry for _seq, entry in merged]


@dataclass
class AuthState:
    vault: TokenVault
    limiter: RateLimiter = field(default_factory=RateLimiter)
    audit: AuditLog = field(default_factory=AuditLog)


# A year. Milestone 6a's spec is that a paired device STAYS paired: a session
# cookie un-pairs the phone the moment its browser closes while the device
# record lives on in devices.json, so the two disagree and the person re-pairs
# forever for no security gain -- training her to leave a QR screen open,
# which is worse than whatever a long-lived cookie costs back.
#
# What it would ordinarily cost is that the issuer can never take it back.
# That does not hold here: `TokenVault.verify()` re-reads and re-parses
# devices.json from disk on *every* call (see vault.py), so a revoked device
# is refused on its very next request no matter how long its cookie claims to
# live, in this process or any other. The `max_age` below is a hint to the
# browser about how long to keep offering the cookie back, never a promise
# about how long it is honoured -- that promise is `verify()`'s alone, and it
# is immediate.
_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365


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

    `max_age` is `_COOKIE_MAX_AGE_SECONDS`, a year, on every policy: without
    one this is a *session* cookie, and closing the browser would un-pair the
    phone while its `Device` record stays in the vault -- the credential and
    the record disagree, and she re-pairs for no security gain. See
    `_COOKIE_MAX_AGE_SECONDS` above for why a year is safe to hand out:
    revocation is enforced by `TokenVault.verify()` re-reading disk on every
    request, not by the cookie's own expiry, so this is never the thing
    standing between a revoked device and this daemon.
    """
    return {
        "httponly": True,       # an XSS in the served UI cannot read it
        "samesite": "strict",
        "secure": policy.secure_cookie,
        "path": "/",
        "max_age": _COOKIE_MAX_AGE_SECONDS,
    }


def _cookie_name_occurrences(connection: HTTPConnection, name: str) -> int:
    """How many times `name` is presented in the raw `Cookie` header(s).

    `connection.cookies` cannot answer this. It is Starlette's `cookie_parser`,
    which collapses a repeated name last-wins
    (`cookie_parser("tenka_device=GOOD; tenka_device=EVIL")` ->
    `{"tenka_device": "EVIL"}`), so the parsed mapping has no way to express
    "how many". Counting needs the unparsed header, and `getlist` because a
    client may also split its jar across several `Cookie` headers.

    Split on `;` and compare the name half, which is exactly the boundary
    `cookie_parser` itself uses -- so this counts the same morsels the parser
    would have collapsed, never more.
    """
    seen = 0
    for header in connection.headers.getlist("cookie"):
        for morsel in header.split(";"):
            key, sep, _value = morsel.partition("=")
            if sep and key.strip() == name:
                seen += 1
    return seen


def cookie_credential(connection: HTTPConnection) -> str:
    """The cookie's value, or `""`. Separate from `credential_from` because
    *which channel* the credential arrived on decides whether CSRF applies."""
    # More than one `tenka_device` presented on a request is never a browser
    # doing its job, and it is refused rather than resolved. Cookies ignore
    # ports, so any page on another port of this host can write one with
    # `path=/`; and `Domain=` from a sibling under a shared parent -- `*.ts.net`
    # or `*.trycloudflare.com`, the parents 6b publishes under, where the
    # neighbours are other people's machines -- plants one inward. RFC 6265
    # s5.4 serialises equal-path cookies oldest-first, so the *attacker's*
    # later-set duplicate is the one a last-wins parser adopts: the operator's
    # browser is silently moved onto a session the attacker also holds, and
    # everything she then does through Studio is readable by it.
    #
    # `__Host-` is the browser-side fix, it *is* adopted now, and it is what
    # actually closes this -- see `HOST_COOKIE_NAME` for the full argument and
    # for why the duplicate rule alone was half a fix in both directions (a
    # permanent lockout on one side, the original fixation untouched on the
    # other). The names are tried in preference order, so a cookie this daemon
    # set over TLS on this exact host beats anything a neighbour planted.
    #
    # The duplicate rule stays, per name, for the listener that cannot use the
    # prefix. Refusing the pair, rather than picking the first, is deliberate:
    # "which of these two did the operator mean?" is not a question this daemon
    # can answer, and guessing right is worth less than never guessing. Logged,
    # because more than one of these is never a browser doing its job and the
    # operator has no other way to find out it happened.
    for name in _COOKIE_NAMES:
        seen = _cookie_name_occurrences(connection, name)
        if seen == 0:
            continue
        if seen > 1:
            logger.warning(
                f"[API] refusing a request presenting {seen} {name} cookies; "
                f"one of them was not set by this daemon")
            return ""
        value = (connection.cookies.get(name) or "").strip()
        if value:
            return value
    return ""


def cookie_name_for(policy: ListenerPolicy) -> str:
    """Which cookie name this listener writes.

    Paired with `cookie_kwargs`, which supplies the three attributes the
    `__Host-` prefix requires -- `secure`, `path="/"`, and no `domain` -- and
    is the reason the prefixed name can be used at all wherever
    `policy.secure_cookie` holds. A browser silently discards a `__Host-`
    cookie that fails any of those, so the two functions have to agree; they
    are kept adjacent and read the same field so they cannot drift.
    """
    return HOST_COOKIE_NAME if policy.secure_cookie else COOKIE_NAME


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


# ─── which hostnames are ours *right now* ────────────────────────────────
class PublishedHosts:
    """The hostnames a **currently running** transport answers on.

    A plain `set` was the wrong shape for this, and the reason is a lifecycle
    rather than a bug in any one line: a transport publishes its public name
    when it starts, and nothing anywhere could take that name back out again.
    `HostGate` and `endpoint_origins()` both read this collection, so a name
    that was published once stayed both an accepted `Host` and a trusted
    `Origin` for the rest of the process's life -- long after the tunnel that
    owned it had stopped.

    That is not merely untidy. A `*.trycloudflare.com` name is assigned by
    Cloudflare for the lifetime of one tunnel and handed to somebody else
    afterwards. The device cookie is host-only and lives a year, so the
    browser keeps attaching it to that hostname no matter who is answering
    there now -- `httpOnly` stops script from reading the cookie, not the
    browser from sending it. A stale entry here is therefore a live session
    credential pointed at a stranger's access log, with no attack beyond
    waiting.

    So membership is derived from ownership, not accumulated: every hostname
    belongs to the transport session that published it, and a transport's
    hostnames disappear the moment that session is retracted.

    **Ownership answers *when* a name is trusted. The listener answers
    *where*, and Milestone 6b is where the second question starts having more
    than one answer.** Until 6b there was a single socket, so "which listener
    published this?" had one answer and storing it would have been ceremony.
    With four sockets serving one app it is the difference between two
    transports and one: a name published by the Cloudflare quick tunnel was
    otherwise a trusted `Host` -- and, through `endpoint_origins()`, a trusted
    `Origin` -- on the funnel port, which is a different transport with a
    different ceiling and a different adversary. So an entry is a
    `(listener, hostname)` pair, and every read is scoped to one listener.

    There is deliberately **no unscoped read**. `__contains__` and `__iter__`
    were both removed rather than merely left unused, because `if host in
    published` is the shape a reader reaches for first and it is exactly the
    bug: it answers "is this one of ours?" when the only question worth asking
    is "is this one of *this listener's*?". `hosts_for(listener)` is the whole
    read surface.
    """

    def __init__(self) -> None:
        # owner -> {(listener port, hostname, public port)}. Keyed by owner
        # because a session is what a name's lifetime hangs off; the
        # listener travels with the name because it is what a *read* is
        # scoped by. The public port travels with it too (fix for the 6b
        # live-test defect below) -- `None` when the publisher did not
        # supply one (test scaffolding, `add()`), which `origins_for` reads
        # the same way it reads an explicit `443`: no port suffix.
        self._by_owner: dict[str, set[tuple[int, str, int | None]]] = {}

    # ─── ownership ──────────────────────────────────────────────────────
    def publish(self, hostname: str, *, owner: str, listener: int,
                public_port: int | None = None) -> None:
        """Trust `hostname` on the listener bound to `listener`, for as long
        as `owner` is running.

        `owner` is any label the transport picks for one *session* of itself
        -- not the transport's type. Two successive runs of the same tunnel
        get different names from the provider, so they must be different
        owners; otherwise stopping the second would leave the first's name
        behind, which is the whole failure this class exists to prevent.

        `listener` is the local port that session's socket is bound to -- the
        same key `policy_for_port` resolves a policy from, so a name and the
        permissions it can reach are scoped by the one piece of addressing a
        client cannot forge.

        `public_port` is the *public* port this listener's tunnel is reached
        on -- `TransportAdapter.public_port()`, e.g. `8443` for `tailnet` --
        carried alongside the hostname purely so `origins_for` can build a
        port-correct `Origin` without this module importing `transports/`
        (which it must not: `io/api/` layering, and `transports/manager.py`
        already imports `unpublish_host` from here, so the reverse import
        would cycle). `None` when the caller has no public port to give
        (test scaffolding, `add()`) -- `origins_for` then omits the port
        exactly as it would for an explicit `443`.

        Normalised through `hostname_of`, which is what `host_is_allowed`
        compares an incoming `Host` with. Both sides therefore agree by
        construction rather than by two spellings of "lowercase and strip"
        happening to stay in step: a name published as `Host.Example:8443`
        and a header reading `host.example` are the same name here. The
        adapter contract already obliges a bare hostname and the mismatch
        failed *closed*, so this removes a way to be surprised rather than a
        live hole -- but a gate whose two halves normalise differently is a
        gate waiting for the day one of them is edited.

        **`public_port` never reaches `hostname_of` or the stored name
        itself** -- only `hostname` does. `hosts_for` and the `Host` gate
        keep reading a bare hostname exactly as before; the port is a third,
        separate element of the stored tuple, read only by `origins_for`.
        """
        name = hostname_of(str(hostname))
        if not name:
            return
        port = None if public_port is None else int(public_port)
        self._by_owner.setdefault(str(owner), set()).add(
            (int(listener), name, port))

    def unpublish(self, owner: str) -> frozenset[str]:
        """Withdraw everything `owner` published. Returns the hostnames that
        were withdrawn.

        Idempotent: an owner that published nothing, or has already been
        withdrawn, is not an error -- a transport's stop path must be safe to
        run twice (a crash handler and an orderly shutdown both call it).
        """
        return frozenset(name for _listener, name, _public_port
                         in self._by_owner.pop(str(owner), set()))

    def owners(self) -> frozenset[str]:
        return frozenset(self._by_owner)

    # ─── the read surface the gates use ─────────────────────────────────
    def add(self, hostname: str, *, listener: int) -> None:
        """A hostname with no transport behind it -- test scaffolding and the
        console, which have no session to tie a lifetime to.

        Owned by its own name, so it is still removable (`unpublish(name)`)
        rather than being the permanent entry this class was built to
        abolish. Production transports call
        `publish(..., owner=..., listener=...)` with a real session id.

        `listener` is required here too, and deliberately has no default:
        "which socket is this name for?" has no sensible answer to guess at,
        and a default would put every scaffolding name on one listener that
        the next reader would then have to discover was arbitrary.

        The owner is the normalised name itself, through the same
        `hostname_of` `publish` uses, so `unpublish(hostname_of(name))`
        removes what `add(name)` created however the caller spelled it.
        """
        self.publish(hostname, owner=hostname_of(str(hostname)),
                     listener=listener)

    def hosts_for(self, listener: int | None) -> frozenset[str]:
        """Every hostname currently published *on this listener*, bare --
        never a port.

        The only read `host_is_allowed` (the `Host` gate, KI-17's load-
        bearing layer) uses, and it must stay exactly this shape: a tunnel
        forwards the public authority in `Host`, and matching it against a
        bare, port-stripped name is the whole point (`hostname_of` on both
        sides -- see `publish`'s docstring). `origins_for` below is the
        *different* question ("what URL reaches this listener"), answered
        separately so this one is never tempted to grow a port.

        `listener` is `None` when the accepting port could not be
        determined at all, and that answers empty rather than everything: a
        connection nobody can place is a connection nothing was published
        for.
        """
        if listener is None:
            return frozenset()
        port = int(listener)
        return frozenset(name for pairs in self._by_owner.values()
                         for entry_port, name, _public_port in pairs
                         if entry_port == port)

    def origins_for(self, listener: int | None) -> frozenset[str]:
        """Every reachable `https://` origin for a hostname published *on
        this listener*, each carrying that listener's own public port when
        it is not the HTTPS default.

        The fix for the 6b live-test defect: `tailnet` publishes on `8443`,
        not 443, so a browser's own `Origin` header for a page served there
        reads `https://<host>:8443` -- and `endpoint_origins` (which trusts
        this set as a front door) and `_endpoints()` (which offers a QR
        built from it) both need the identical string, or the two disagree
        about what "this listener" is reachable at. `hosts_for` above stays
        bare on purpose (the `Host` gate's job); this is the separate
        question of what URL a client actually reaches this listener
        through, so it lives here as a second method rather than a second
        meaning for the first one.

        A stored `public_port` of `None` (nothing was supplied at publish
        time -- test scaffolding, `add()`) is treated exactly like an
        explicit `443`: no port suffix, the same default-port omission
        `TransportAdapter.public_url` applies for `funnel` and `quick`. A
        browser never puts the default port in `Origin` either, so emitting
        one here would refuse a *legitimate* funnel origin -- the opposite
        direction of this fix.
        """
        if listener is None:
            return frozenset()
        port = int(listener)
        origins: set[str] = set()
        for pairs in self._by_owner.values():
            for entry_port, name, public_port in pairs:
                if entry_port != port:
                    continue
                if public_port is None or public_port == 443:
                    origins.add(f"https://{name}")
                else:
                    origins.add(f"https://{name}:{public_port}")
        return frozenset(origins)

    def __len__(self) -> int:
        """How many distinct `(listener, hostname)` pairs are live.

        A count, never a membership test: nothing decides trust from this, and
        it exists so an operator-facing summary can say how many names are
        currently published without being handed the unscoped read above.
        Counted on `(listener, hostname)` alone, ignoring the public port --
        this stays a count of *names*, the same promise it made before the
        public port existed on the stored tuple at all.
        """
        return len({(entry_port, name) for pairs in self._by_owner.values()
                    for entry_port, name, _public_port in pairs})


def unpublish_host(app_state, owner: str) -> frozenset[str]:
    """Stop trusting everything `owner` published, on this app's state.

    The counterpart a transport calls on its own stop path. Kept as a
    module-level function beside `host_is_allowed` and `endpoint_origins` --
    the two gates that read the collection -- so "what makes a hostname
    trusted" and "what takes that back" are one thing to find, not two.
    """
    published = getattr(app_state, "published_hosts", None)
    if isinstance(published, PublishedHosts):
        return published.unpublish(owner)
    return frozenset()


# ─── DNS rebinding: the Host allow-list ──────────────────────────────────
def hostname_of(value: str) -> str:
    """The host part of a `Host` header or an origin authority, lowercased."""
    value = value.strip().lower()
    if value.startswith("["):                 # [::1]:8787
        end = value.find("]")
        return value[:end + 1] if end != -1 else value
    return value.split(":", 1)[0]


def host_is_allowed(host_header: str | None, published: PublishedHosts, *,
                    port: int | None, policy_name: str | None) -> bool:
    """A rejection gate, never a selection one -- scoped to one listener.

    A page on `evil.example` can re-resolve its own hostname to 127.0.0.1 and
    then speak to this daemon as same-origin -- the browser's origin checks
    all pass, because as far as it knows nothing changed. What does not change
    is the `Host` header it keeps sending: still `evil.example`, because that
    is the name in the address bar. Refusing every name that is not one of
    ours is the standard defence, and it works precisely because the attacker
    cannot alter that header from a page.

    **`local` never accepts a published name, and that clause is KI-17's
    layer 3.** `policy_for_port` keys on the accepting port, which is correct
    and unforgeable -- but a tunnel pointed at the *existing* Studio port
    resolves to `POLICIES["local"]`: admin, bearer, and a ceiling holding
    `EXECUTE` and `SYSTEM_CONTROL`. Every ceiling Milestone 6a.5 built is
    bypassed, by the obvious implementation rather than by a bug. The way out
    is that a tunnel cannot hide what it is: `tailscale serve` and
    `cloudflared` both forward the **public** authority in `Host`, so such a
    request arrives on the local port carrying `something.ts.net` or
    `something.trycloudflare.com` -- and is refused here, before
    authentication, before policy lookup, before any route runs. Two other
    layers (TENKA builds every tunnel's argv; a preflight refuses a stale
    `tailscale serve` mapping) are procedural and assume the tunnel is one
    TENKA launched or can see. This one holds against a tunnel TENKA never
    launched and knows nothing about, and it is the only one that does. If it
    is ever relaxed, KI-17 is live again.

    **The stated gap**, recorded rather than claimed away:
    `cloudflared --http-host-header 127.0.0.1:8787` rewrites `Host` to a
    loopback name and defeats this. That requires an attacker already
    executing processes on this machine, at which point the local listener is
    not the weakest thing available to them -- and layer 2 catches the
    honest-mistake version of the same configuration. Do not try to build
    around it here; there is nothing left in the request to distinguish it
    with.

    Loopback names are allowed on **every** listener, including one whose port
    nobody declared. A local process reaching a transport port gains strictly
    less than reaching the local port, which it can already reach, so refusing
    them would cost the loopback health check and buy nothing. An unregistered
    port therefore passes this gate for a loopback name and is answered 401 by
    `authenticate()`, which is the refusal that has always covered it -- but
    it accepts no *published* name at all, because "nobody declared what this
    socket is" is not a listener a name can belong to.

    That reading of "an unknown port refuses" -- refuse its published names,
    not its loopback ones -- was **reviewed and ratified**, and is recorded
    here so it is not "repaired" back. Spec §2.4 item 4 re-pins 401 for an
    unregistered port and
    `test_api_cookie_auth.py::test_a_request_on_an_unregistered_port_is_refused`
    asserts it; answering 421 here instead would have moved an existing
    assertion, and it would have bought nothing against KI-17, whose whole
    premise is a tunnel aimed at a port that very much *is* registered.
    `tests/test_6b_host_scoping.py::test_dropping_a_listener_from_the_registry_
    stops_its_names_immediately` pins both halves of the behaviour together.

    The port in `Host` is ignored. It carries no security meaning here (the
    real port is `port`, the one the socket was accepted on) while insisting
    on it would reject every tunnel: a transport forwards with the public
    authority, whose port is not this daemon's.
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
    if policy_name is None or policy_name == "local":
        return False
    return name in published.hosts_for(port)


# ─── the active endpoint set: what may talk to us ────────────────────────
def endpoint_origins(app_state, port: int | None,
                     policy: ListenerPolicy) -> frozenset[str]:
    """Every origin this daemon considers to be one of its own front doors.

    Built from the port the connection was accepted on (so a second listener
    cannot lend its origins to the first), plus whatever hostnames a running
    transport has published **on that same port**, plus -- on a `local`
    listener only -- the configured development origins. Studio runs on its
    own dev server during development and is served same-origin in production;
    letting the dev list through on a tunnelled listener would mean a public
    URL trusting an origin that only ever existed on someone's laptop.

    The published half used to read the whole collection, which meant the
    `port` argument only scoped the loopback pair while the tunnel names it
    sat next to were shared by every listener -- the funnel port trusting a
    quick tunnel's origin was the exact thing `port` was added to prevent.
    `hosts_for(port)` is now the only read, and a `None` port contributes no
    published name at all: a connection whose accepting port could not be
    determined belongs to no listener, so no listener's names are its own.

    **`local` contributes no published name either, and that clause exists to
    agree with `host_is_allowed`.** The two functions answer different
    questions -- "may this name reach us?" and "is this origin one of our own
    front doors?" -- but they answer them about the same listener, and a
    reader who learns the rule from one will assume it holds in the other.
    Left disagreeing, the local listener refused `https://tunnel.ts.net` as a
    `Host` while trusting it as an `Origin`: a page there could drive
    `http://127.0.0.1:<local port>` cross-origin, since that request carries a
    loopback `Host` the gate allows and an `Origin` this set would have
    vouched for. Nothing legitimate needs it -- Studio served over a tunnel
    makes its requests to the tunnel listener, never to the loopback one -- so
    the rule is the same on both sides: a name is trusted only where it was
    published, and `local` publishes nothing.
    """
    origins: set[str] = set()
    published = getattr(app_state, "published_hosts", None)
    if port is not None:
        origins.add(f"http://127.0.0.1:{port}")
        origins.add(f"http://localhost:{port}")
        if policy.name != "local" and isinstance(published, PublishedHosts):
            # `origins_for`, never `hosts_for` -- a published transport
            # hostname is always reached over TLS (both Tailscale and
            # Cloudflare terminate https for it), but `tailnet` is reached
            # on a non-default port (8443) that a bare `https://{host}`
            # omits. A browser's own `Origin` header for a page served over
            # `tailnet` carries that port, so trusting the portless form
            # here refused every real one (`_CROSS_SITE`, live-tested):
            # this must build the identical string the browser sends, and
            # `origins_for` is where that construction is shared with
            # `_endpoints()` (`routes/pairing.py`) rather than repeated.
            origins.update(published.origins_for(port))
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


def anonymous_key(scope) -> str:
    """The limiter key spent by a caller that did not verify to a device.

    The accepting port, and nothing the caller sent. This used to be
    `request.client.host`, defended by the claim that "`cloudflared` and
    `tailscale funnel` both connect from 127.0.0.1, so `source` collapses to a
    single shared key for every remote caller". That claim was false in the
    deployed configuration: uvicorn's `proxy_headers` default installs
    `ProxyHeadersMiddleware`, which rewrites `scope["client"]` from
    `X-Forwarded-For` for exactly the loopback peers the claim names -- so the
    anonymous budget *and* the exponential lockout were both keyed on a value
    the client chose. Ten wrong guesses under one header value, then a
    different value, and the next request was answered as a fresh, unmetered
    caller.

    `server.py` now turns that middleware off, which makes the old key honest
    again. This function is the other half, and the more durable one: rather
    than depending on one uvicorn argument staying right forever, the key is
    derived from the one piece of addressing a client cannot choose at all --
    the local port the connection was accepted on, the same field `policy.py`
    already trusts to decide what a request may do. Put a real reverse proxy
    in front of this daemon tomorrow, or re-enable proxy headers by accident,
    and the budget still cannot be rotated.

    What it gives up is fairness *between* anonymous callers on one listener,
    and that costs nothing worth keeping. A caller that verifies is metered by
    `device_key()` and never reaches this, so no honest device shares a budget
    with an attacker; what remains here is a wrong token, a missing one, or a
    request that could not be attributed at all. `routes/pairing.py` has
    metered its whole unauthenticated route on one fixed key since Task 10,
    for the same reason and with the same trade written out.
    """
    port = accepting_port(scope)
    return f"anon:{port}" if port is not None else "anon:unknown"


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
    identity to meter, so it is charged against one shared per-listener key
    instead (`anonymous_key()`, the accepting port -- never the client
    address, which a proxy header could rewrite). Verifying before checking
    either budget also means a valid token is never refused a 429 it never
    earned just because other traffic already burned the anonymous budget.

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

    That collapse used to be an *assumption*, and a false one. `source` was
    `request.client.host`, and uvicorn installs `ProxyHeadersMiddleware`
    whenever `proxy_headers` is true (its default), which rewrites
    `scope["client"]` from `X-Forwarded-For` for any peer inside
    `forwarded_allow_ips` -- default `127.0.0.1`, precisely the peer address
    every tunnel here connects from. So the anonymous budget and the
    exponential lockout were both keyed on a value the caller supplied: ten
    wrong guesses under one header, a different header, and the next request
    was answered as a fresh, unmetered caller. Task 10's review "proved"
    otherwise through `TestClient`, which speaks ASGI directly and never
    installs uvicorn's middleware at all.

    It is no longer an assumption. `server.py` passes `proxy_headers=False`
    explicitly, and `source` above is `anonymous_key()` -- the accepting port,
    the one piece of addressing a client cannot choose -- so the collapse is
    what the code *does* rather than what the deployment is hoped to produce.
    `RateLimiter` is bounded too (see `_prune`), so the unbounded-growth half
    of the same mistake cannot come back on its own either.

    The `Device` this returns is the device *as this listener sees it*: its
    grants have already been intersected with the listener's ceiling, so every
    caller downstream -- `require()`, and `routes/commands.py`, which checks a
    grant of its own choosing against `device.grants` without going through
    `require()` at all -- inherits the narrowing for free. Doing it here rather
    than inside `require()` is the difference between a ceiling that applies
    to every route and one that applies to the routes that remembered to ask.
    """
    state: AuthState = request.app.state.auth
    # Never `request.client.host`: see `anonymous_key()` for why the caller
    # must not get to pick which budget it spends.
    source = anonymous_key(request.scope)

    policy = policy_for_scope(request.scope, request.app.state.listener_policies)
    if policy is None:
        # Nobody declared what this socket is, so it carries nothing. Failing
        # closed here is the whole point of keying policy on the port: a
        # listener added in a later milestone and forgotten in the registry
        # serves 401s, rather than quietly inheriting loopback's rights.
        if not state.limiter.check(source):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="too many requests")
        raise UNAUTHORIZED
    request.state.policy = policy

    # Before `verify()`, deliberately, and matching the order the event socket
    # already used. Run afterwards, these two checks answered 403 to a
    # cross-site request holding a valid cookie and 401 to the same request
    # without one -- a "does this browser hold a working credential for this
    # daemon?" oracle that any page could query. Neither check needs to know
    # whether the credential is good, so neither should be able to reveal it.
    _refuse_cross_site(request, policy)

    token = credential_from(request, policy) or ""
    # Off the event loop, for the reason `touch()` below already is.
    #
    # `verify()` reads `instance_secret` and re-reads and re-parses
    # `devices.json` from disk, then HMACs every record, and it runs on every
    # request before anything has metered it -- so an unauthenticated caller
    # presenting any cookie at all sets the rate. That is deliberate and stays
    # deliberate: re-reading disk every single call is the entire reason a
    # one-year cookie is defensible, because it is what makes revocation
    # immediate (see `_COOKIE_MAX_AGE_SECONDS`).
    #
    # What was not deliberate is that the reads were *synchronous on the loop
    # the assistant herself runs on*. The measurement this file already
    # records for `touch()` -- eight unrelated requests 1.5x-7.7x slower while
    # one blocking call was in flight -- is the same cost, and here it is paid
    # by traffic nobody has authenticated yet. `asyncio.to_thread` turns
    # "everything stops" into "this request waits", which is what a flood
    # should cost: the flooder's own latency, not the daemon's.
    #
    # A reviewer proposed instead moving `state.limiter.check(source)` above
    # this line so the anonymous window bounds the read *rate*. That was
    # tried and reverted, and the reason is worth writing down so it is not
    # tried again. `source` is `anonymous_key()` -- the accepting port -- and
    # this file's own docstring explains why: a tunnel connects from
    # 127.0.0.1, so every remote caller collapses onto one shared key. A
    # source-keyed gate ahead of verification therefore cannot tell a flood
    # from a paired device; refusing on it means an anonymous flood, or ten
    # wrong guesses earning the source's exponential lockout, denies every
    # paired device on that listener. Measured, not reasoned: with the check
    # moved, `test_a_valid_token_is_never_refused_by_an_exhausted_anonymous_
    # window` and `test_a_valid_device_survives_a_flood_of_wrong_tokens_from_
    # its_own_source` both fail. Those two are the property Task 10 landed
    # this ordering for. A rate limiter keyed on a value the attacker shares
    # with its victims is a weapon handed to the attacker, and the vault read
    # is not expensive enough to buy one with.
    device = await asyncio.to_thread(state.vault.verify, token)

    if device is not None:
        key = device_key(device)
        if not state.limiter.check(key):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="too many requests")
        state.limiter.record_success(key)
        # Captured before the narrowing below overwrites `device.grants` with
        # the effective set. GET /v1/session needs both: what the device was
        # issued, and what this listener lets through. Stashing the pre-narrow
        # set here -- rather than having the route re-verify the token or call
        # `vault.devices()` itself -- means there is exactly one vault read per
        # request and exactly one place (`effective()`, right below) that ever
        # narrows a grant set.
        issued = device.grants
        # Milestone 6b's third bound, and the only one that moves: a live,
        # expiring, per-device ceiling raise (`raises.py`), folded into
        # `effective()` as its third argument. It can only widen the
        # *transport* side of the intersection, and only within that policy's
        # fixed `raisable` -- `issued` above is untouched, so a raise can never
        # manufacture a grant the device was not given.
        #
        # `getattr`, because `app.state.raises` is set by `create_app` in a
        # sibling task: an app built before that lands (or a test that never
        # attaches a store) has no attribute at all, and `frozenset()` is both
        # the correct fail-closed answer and exactly 6a.5's behaviour.
        #
        # Off the event loop, and its own hop rather than folded into the
        # `verify()` call above. Two things want it that way. `RaiseStore`
        # takes a `threading.Lock` that the admin raise route writes under, and
        # the loop this daemon shares with the assistant must not wait on a
        # lock held by an unrelated request. And the vault read above has to
        # stay spelled `asyncio.to_thread(state.vault.verify, ...)`: the 6a.5
        # availability pass pins that literal dispatch by reading this
        # function's own source, so wrapping the two in one worker callable
        # would respell the guard rather than keep it. The extra dispatch buys
        # both, and it is dwarfed by the devices.json read and HMAC this same
        # request already paid for.
        raise_store = getattr(request.app.state, "raises", None)
        raised: frozenset[Capability] = frozenset()
        if raise_store is not None:
            raised = await asyncio.to_thread(
                raise_store.capabilities_for, device.device_id, policy.name)
        grants = effective(issued, policy, raised)
        if not grants:
            # This transport carries nothing this device holds. Refused as an
            # authentication failure, not a 403: a device that authenticates
            # with an empty grant set can still reach any route gated on
            # `authenticate` alone, which turns the 404-vs-403 split in
            # `run_command` into an oracle for which command ids exist.
            # `TokenVault.issue()` refuses an empty grant set at the source
            # for exactly this reason; the ceiling is the other way to arrive
            # at one, and it is closed the same way.
            raise UNAUTHORIZED
        device = replace(device, grants=grants)
        request.state.device = device
        request.state.issued_grants = issued
        # Spec §3.6: an audit event on every request a raise put a capability
        # within reach of -- the difference between knowing a door was unlocked
        # and knowing somebody was inside while it was. A seven-day window is
        # not something to hold in your head, and the mint line alone says only
        # that it opened.
        #
        # **Read what this records, not what it sounds like.** `applied` is a
        # function of the raise and the policy alone -- it never consults what
        # the request went on to require -- so *every* authenticated request on
        # the raised listener writes one of these entries while the raise is
        # live: a Studio status poll, a `GET /v1/session`, anything. The entry
        # says which capabilities the raise put in reach on this request, never
        # which one the handler actually spent; the method and path beside it
        # are what narrow that down. `test_a_raise_in_reach_records_on_every_
        # request_not_only_the_one_that_spends_it` pins exactly that, so nobody
        # has to infer the semantics from this comment.
        #
        # Recorded here rather than inside `require()`, which is where a
        # per-capability hook -- one that *could* tell spending from reach --
        # would naturally go. `require()` does not see the consumption that
        # matters most: `POST /v1/chat` is gated on CHAT_SEND and then hands
        # `device.grants` straight to the assistant, where the intent gate
        # spends EXECUTE with no `require()` anywhere in sight, and
        # `run_command` checks a grant of its own choosing inline. Hooking
        # `require()` plus each of those by name is the enumerate-every-path-
        # around-the-boundary problem this milestone has already lost to twice.
        # Every path comes through this function, so this is the one site that
        # cannot be walked around, and over-recording is the fail-safe
        # direction to be wrong in.
        applied = grants - effective(issued, policy)
        if applied:
            state.audit.record(AuditEntry(
                at=datetime.now(timezone.utc).isoformat(),
                device_id=device.device_id,
                method=request.method,
                path=redact_secrets(request.url.path),
                outcome=f"raised:{'+'.join(sorted(c.value for c in applied))}",
            ))
        # Last on the success path, and only on it. "When was this device last
        # used?" is the column the revoke list is read by -- a row nobody
        # recognises with a timestamp from three months ago is what makes
        # somebody click revoke -- and it is a fact about a *verified* device,
        # so a wrong token has nothing to attribute. A device refused further
        # down for lacking a capability was still that device, though, which
        # is why this sits here rather than inside `require()`.
        #
        # The write is throttled to one per device per minute inside the vault
        # (see `TokenVault.touch`), because `_save` costs a full JSON rewrite
        # plus an `icacls` subprocess. What is NOT throttled is the read the
        # throttle check itself needs: this adds one `devices.json` load and
        # parse to every authenticated request, on top of the one `verify()`
        # above already did. That is the standing cost of the column, paid on
        # the request path by design rather than hidden behind a second cache
        # here that could disagree with the vault's own window.
        #
        # `touch()` now raises `VaultUnavailableError` (either half: a read
        # that could not determine the document's real content, or a write
        # that could not land) instead of silently no-op'ing when devices.json
        # is locked by a scanner, a backup tool, or a second TENKA process --
        # see `TokenVault.touch`'s own docstring. That is the right behaviour
        # for the vault itself, which must never guess at a document it
        # cannot read or silently swallow a write that did not happen, but
        # this call site is not the place to turn either into a failed
        # request: the device already verified above, off a devices.json read
        # that plainly succeeded moments earlier, and every one of the checks
        # that must not be bypassed (capability, listener ceiling, rate limit)
        # has already passed. Failing an otherwise-fully-authenticated request
        # because a best-effort "last seen" bookkeeping write hit a transient
        # lock would make the revoke-list column more expensive to maintain
        # than the thing it protects -- and letting the write half propagate
        # unwrapped would be worse than a failed request: `PermissionError`
        # (`VaultWriteError`'s underlying cause) is mapped by `errors.py` to
        # 403 "protected path", so an uncaught write failure here would answer
        # 403 -- "you are not allowed" -- to every authenticated request for
        # as long as the lock held, which is a lie this call site must not
        # tell. Best-effort either way: log it, keep going.
        #
        # Off the event loop, always. `touch()` is synchronous and its write
        # half spawns `icacls` through `subprocess.run(..., timeout=10)` --
        # measured at 13-90ms of *blocking* wall time per call. This daemon
        # shares its loop with the assistant, so those milliseconds are never
        # this one request's alone: eight unrelated, unauthenticated requests
        # measured 1.5x-7.7x slower while a single such write was in flight,
        # because the loop's one thread was sitting inside a subprocess wait
        # rather than servicing anybody. `asyncio.to_thread` turns
        # "everything stops" into "this request waits", and it is safe
        # exactly here: `TokenVault` guards the whole load-check-save
        # sequence with a `threading.Lock`, which is the case that lock was
        # written for.
        try:
            await asyncio.to_thread(state.vault.touch, device.device_id)
        except VaultUnavailableError as exc:
            logger.warning(f"[API] could not record last-seen for "
                           f"{device.device_id}: {exc}")
        return device

    if not state.limiter.check(source):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="too many requests")
    if token:
        state.limiter.record_failure(source)
    raise UNAUTHORIZED


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
    refuse_unknown_origin(request, policy)
    if request.method.upper() in _AMBIENT_METHODS and cookie_credential(request):
        if not request.headers.get(CSRF_HEADER):
            raise _CSRF_MISSING


def refuse_unknown_origin(request: Request, policy: ListenerPolicy) -> None:
    """The `Origin` half of the gate above, on its own.

    Split out for `POST /v1/pair`, the one route that needs this half and
    must not have the other. That route reads no credential at all -- its
    authority is the pair code in the body, which a cross-site page would
    have to supply itself -- so there is no ambient authority for a CSRF
    header to defend, and demanding one would refuse a phone whose stale
    cookie from a revoked pairing is still in its jar. The `Origin` check
    still earns its place there: it stops a page the user happens to be
    visiting from redeeming a code it learned and planting the resulting
    device cookie in her browser.

    Only checked when the header is present. A browser omits `Origin` on a
    same-origin GET and on a plain navigation, and a non-browser client never
    sends one, so demanding it would break both the UI this daemon serves and
    every script on loopback.
    """
    origin = request.headers.get("Origin")
    # `is not None`, not truthiness. A *present but blank* `Origin` -- `""` or
    # `"   "` -- is malformed input, and truthiness posted it into the
    # absent-header branch, which on `local` means allow and on `POST /v1/pair`
    # means redeem. Nothing legitimate is lost by refusing it: a browser sends
    # a serialised origin, `null`, or no header at all, and `fetch` cannot set
    # the header from script (forbidden header name), so the value is only
    # reachable from a non-browser client -- which sends no header at all and
    # keeps the `None` branch below. The blank is checked explicitly rather
    # than left to the allow-list lookup so that a stray empty entry in the
    # configured development origins could never make it match.
    if origin is not None:
        if not origin.strip() or not origin_is_known(
                origin, request.app.state,
                accepting_port(request.scope), policy):
            raise _CROSS_SITE


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


def admin_capability_satisfied(capability: Capability,
                                grants: frozenset[Capability],
                                policy: ListenerPolicy) -> bool:
    """The exact condition `require_admin(capability)` enforces over HTTP --
    `capability` held AND a listener trusted to manage this daemon's own
    credentials (`policy.admin`) -- exposed as a plain predicate rather than
    only as a FastAPI dependency.

    `events.py`'s collection-scoped invalidate fanout is the reason this is
    a standalone function: a socket is not a route, so it cannot go through
    `Depends(require_admin(...))`, but "would this device's next GET to the
    admin route this resource lives behind succeed right now?" has to be the
    *same* question, answered from the *same* place -- never a second,
    hand-derived copy of `policy.admin and capability in grants` that could
    drift from this one as either side changes.
    """
    return capability in grants and policy.admin


def require_admin(capability: Capability):
    """Dependency factory: `require(capability)`, then insist on a listener
    that is trusted to manage this daemon's own credentials.

    Two gates, not one, and neither is redundant. `policy.admin` is loopback
    alone, so device enrollment and revocation cannot be reached from a
    tunnel at all -- one compromised remote session must not be able to open
    a second, permanent door, nor to close the doors somebody else is using.
    The capability is what stops the *other* direction: a device paired only
    to watch, sitting on the loopback listener, could otherwise mint itself a
    pair code carrying every capability there is. That is the same escalation
    arrived at from inside the house, and the transport check says nothing
    about it.

    Ordered capability-first so the two refusals are indistinguishable in
    practice as well as in shape: whichever gate a caller trips, it is the
    same 403 body `require()` already returns.
    """

    async def _dependency(request: Request,
                          device: Device = Depends(require(capability))) -> Device:
        # `request.state.policy` is set by `authenticate()`, which
        # `require(capability)` above depends on, so it is always present here.
        # Routed through `admin_capability_satisfied` so this dependency and
        # the event hub's fanout gate are provably the same rule -- see that
        # function's docstring.
        if not admin_capability_satisfied(capability, device.grants,
                                          request.state.policy):
            raise _NOT_ADMIN
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
