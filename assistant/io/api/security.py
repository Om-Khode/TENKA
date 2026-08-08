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

    def check(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now < self.locked_until.get(key, 0.0):
            return False
        window = self.hits[key]
        while window and now - window[0] > _WINDOW_SECONDS:
            window.popleft()
        if len(window) >= _MAX_PER_WINDOW:
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


async def authenticate(request: Request) -> Device:
    state: AuthState = request.app.state.auth
    source = request.client.host if request.client else "unknown"

    if not state.limiter.check(source):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="too many requests")

    device = state.vault.verify(_bearer(request))
    if device is None:
        state.limiter.record_failure(source)
        raise _UNAUTHORIZED

    state.limiter.record_success(source)
    request.state.device = device
    return device


def require(capability: Capability):
    """Dependency factory: authenticate, then insist on one grant."""

    async def _dependency(device: Device = Depends(authenticate)) -> Device:
        if capability not in device.grants:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="capability not granted")
        return device

    return _dependency
