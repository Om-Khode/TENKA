# assistant/io/api/pairing.py
"""In-memory, single-use pair codes for the phone-pairing QR flow.

A pair code is the only thing standing between a stranger and a daemon that
can execute arbitrary code on the machine, so its properties are load-bearing:

- In memory only, never persisted. A three-minute code has no reason to
  survive a restart, and not writing it to disk means there is no file to
  steal.
- Single use, consumed atomically under a lock. A replay must be
  indistinguishable from a wrong code.
- Grants ride on the code, chosen on the laptop before the QR is shown, so
  the pairing request that redeems it can never widen them.
- At most one live code. Minting a new one invalidates the previous, so a
  forgotten QR screen from an hour ago is not still a working credential
  path.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass

from .vault import Capability

# Crockford-ish: no I, L, O, U -- each is easily misread as 1, 1, 0, V when
# copied off a screen by hand, which is the only path a code ever fails on
# (the QR itself carries the exact string; a human never retypes it there).
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# Long enough to type once during a live pairing, short enough that a code
# left on an unattended screen is worthless within a few minutes.
CODE_TTL_SECONDS = 180.0


@dataclass(frozen=True)
class PairCode:
    code: str                        # "7K2M-9QX4"
    label: str
    grants: frozenset[Capability]
    expires_at: float                # time.monotonic() basis


def _generate_code() -> str:
    # 8 symbols over a 32-character alphabet is 5 bits/symbol, ~40 bits total
    # -- plenty for a code that lives at most CODE_TTL_SECONDS and is
    # invalidated by the next mint, so there is no meaningful window for an
    # offline guess, let alone a brute-force one against a live daemon.
    symbols = "".join(secrets.choice(_ALPHABET) for _ in range(8))
    return f"{symbols[:4]}-{symbols[4:]}"


class PairCodeStore:
    """Holds at most one live `PairCode` at a time, in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: PairCode | None = None

    def mint(
        self, label: str, grants: frozenset[Capability], *, now: float | None = None
    ) -> PairCode:
        """Create a new code, replacing (invalidating) any code already live."""
        moment = time.monotonic() if now is None else now
        pair_code = PairCode(
            code=_generate_code(),
            label=label,
            grants=grants,
            expires_at=moment + CODE_TTL_SECONDS,
        )
        with self._lock:
            self._current = pair_code
        return pair_code

    def consume(self, code: str, *, now: float | None = None) -> PairCode | None:
        """Redeem `code` exactly once. Wrong, expired, or already-used all read
        the same to the caller: `None`, never an exception.

        `code` is untrusted input from the wire, so it can be anything --
        including a lone UTF-16 surrogate half (`"\\ud800"`), which
        `str.encode("utf-8")` rejects with `UnicodeEncodeError`, a subclass of
        ValueError. `hmac.compare_digest` needs both sides encoded to bytes to
        compare the actual code characters rather than Python object
        identity/length quirks, so that encode step runs on attacker-chosen
        text and must not be allowed to raise out of this method.
        """
        moment = time.monotonic() if now is None else now
        with self._lock:
            current = self._current
            if current is None:
                return None
            try:
                matches = hmac.compare_digest(
                    current.code.encode("utf-8"), code.encode("utf-8")
                )
            except (UnicodeEncodeError, AttributeError):
                # AttributeError guards a non-str `code` (e.g. None) reaching
                # here from a caller that skipped validation upstream.
                return None
            if not matches:
                return None
            # Consume before checking expiry: an expired code must not be
            # redeemable twice either, and clearing the slot here makes the
            # "already used" and "expired" outcomes collapse into the same
            # single-use guarantee regardless of which check trips first.
            self._current = None
            if moment >= current.expires_at:
                return None
            return current

    def current(self, *, now: float | None = None) -> PairCode | None:
        """Peek at the live code without consuming it, or `None` if there is
        none or it has expired."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            current = self._current
        if current is None or moment >= current.expires_at:
            return None
        return current
