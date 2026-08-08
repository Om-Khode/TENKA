# assistant/io/api/security.py
"""Authentication, capability checks, rate limiting, audit.

Authentication is a router-level dependency rather than a per-route decorator,
so a new route is authenticated by construction. Unknown, malformed and revoked
tokens produce one identical response after the same work.

Layering: io/api — core + config only.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status

from .vault import Capability, Device, TokenVault

logger = logging.getLogger(__name__)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="unauthorized",
    headers={"WWW-Authenticate": "Bearer"},
)

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


def _bearer(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return value.strip()


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
    flood is still bounded -- the sliding window above caps it at roughly
    `_MAX_PER_WINDOW` requests per `_WINDOW_SECONDS`, sustained indefinitely
    -- it simply never escalates into the exponential, multi-minute lockout
    that a wrong-token guesser earns.
    """
    state: AuthState = request.app.state.auth
    source = request.client.host if request.client else "unknown"
    token = _bearer(request)
    device = state.vault.verify(token)

    if device is not None:
        key = device_key(device)
        if not state.limiter.check(key):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="too many requests")
        state.limiter.record_success(key)
        request.state.device = device
        return device

    if not state.limiter.check(source):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="too many requests")
    if token:
        state.limiter.record_failure(source)
    raise _UNAUTHORIZED


def require(capability: Capability):
    """Dependency factory: authenticate, then insist on one grant."""

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
