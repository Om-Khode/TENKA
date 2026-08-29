"""
extension_ws.py — the socket the browser extension dials in on.

The extension is the client and dials out; this is the server. That direction
is not a preference: an extension cannot hold a listening port, and a server
dialling into a browser would have to know which browser and when.

## What authenticates, and in which order

Four checks, cheapest and least trusting first:

1. **protocol version** — no negotiation. A driver that half-speaks a protocol
   fails in the middle of a task, which is worse than not connecting.
2. **`domQuerySha256`** — the extension reports the digest of the query file it
   shipped; it must equal our vendored copy. MV3's CSP forbids sending the file
   over the wire, so two copies exist and are compared rather than trusted.
3. **token** — compared in constant time against the one
   `browser_extension_setup` minted.
4. **`Origin`** — must be `chrome-extension://` or `moz-extension://`.

The order matters in one direction only: **`Origin` is checked before the token
is examined for validity**, so a non-extension client learns nothing about
whether a token it guessed was close. The version and digest checks run first
because they are decidable without any secret at all, and because answering
"your build is wrong" to a client that also has a bad token is the more useful
of the two answers.

## One client, and the first one keeps the socket

A second connection is refused; the existing one stays serving. The reverse —
letting a newcomer displace the incumbent — means anything that can reach the
port can silently take over an in-flight task, and the operator's browser goes
quiet with no error anywhere.

## Authority

This listener's `ListenerPolicy` has an empty ceiling, so no HTTP route on this
port answers at all (`policy.py`). That is not a belt-and-braces measure around
this file; it is the reason this file can hold its own token check without also
becoming a second front door into the API. The extension is a target, not a
principal: it never asks TENKA to run an intent, and nothing here can.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...core import latch_protocol as proto

logger = logging.getLogger("api.extension_ws")

#: Bumped when the on-disk shape changes. An older marker is rejected rather
#: than migrated — a token store this small is cheaper to re-mint than to
#: upgrade, and a half-understood credential file is worse than none.
TOKEN_SCHEMA_VERSION = 1

_DEFAULT_CALL_TIMEOUT = 30.0


class ExtensionTokenError(RuntimeError):
    """The stored extension credential is missing or unreadable."""


# ─── The credential ───────────────────────────────────────────────────────


def token_path(home: Path | None = None) -> Path:
    base = home if home is not None else Path.home() / ".tenka"
    return base / "extension_token.json"


def mint_token(home: Path | None = None) -> str:
    """Create and store a new extension credential, replacing any existing one.

    `token_urlsafe(32)` — 256 bits. The token is pasted by hand into a popup, so
    it is deliberately not longer than that; what protects it is that the socket
    it opens is loopback-only and grants no intent authority, not its length.
    """
    path = token_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    payload = {"schema_version": TOKEN_SCHEMA_VERSION, "token": token}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return token


def read_token(home: Path | None = None) -> str | None:
    """The stored credential, or `None` if there is not a usable one.

    Every failure — absent, unparseable, wrong schema version, empty — returns
    `None`, and `None` refuses. A token store that cannot be read must never
    become a socket that accepts anything.
    """
    path = token_path(home)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema_version") != TOKEN_SCHEMA_VERSION:
        logger.warning(
            f"[LATCH] ignoring extension token with schema_version="
            f"{raw.get('schema_version')!r} (expected {TOKEN_SCHEMA_VERSION}); "
            f"re-run the extension setup to mint a current one"
        )
        return None
    token = raw.get("token")
    return token if isinstance(token, str) and token else None


def clear_token(home: Path | None = None) -> bool:
    """Remove the stored credential. True if there was one to remove."""
    path = token_path(home)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning(f"[LATCH] could not remove {path}: {e}")
        return False


# ─── Handshake decisions, as a pure function ─────────────────────────────


@dataclass(frozen=True)
class HandshakeVerdict:
    """Accept, or refuse with a code and a reason.

    Separated from the socket so the ordering of the checks can be tested
    without a transport. The ordering is the security-relevant part, and a test
    that has to stand up a WebSocket to assert it will be written once and then
    quietly not extended.
    """

    ok: bool
    code: int | None = None
    reason: str = ""


def evaluate_handshake(
    hello: Any,
    *,
    origin: str | None,
    expected_token: str | None,
    expected_digest: str,
    occupied: bool,
) -> HandshakeVerdict:
    """Decide a `hello` frame. Pure; no I/O, no state."""
    if not isinstance(hello, dict) or hello.get("type") != proto.Frame.HELLO:
        return HandshakeVerdict(False, proto.Err.PROTOCOL_MISMATCH, "first frame was not hello")

    # Origin first among the credential-ish checks: a client that is not an
    # extension learns nothing about the token from how we answer.
    if origin is None or not origin.startswith(proto.EXTENSION_ORIGIN_SCHEMES):
        return HandshakeVerdict(
            False, proto.Err.UNAUTHORIZED,
            "origin is not a browser extension",
        )

    version = hello.get("protocolVersion")
    if version != proto.PROTOCOL_VERSION:
        return HandshakeVerdict(
            False, proto.Err.PROTOCOL_MISMATCH,
            f"protocol {version!r} != {proto.PROTOCOL_VERSION}",
        )

    digest = hello.get("domQuerySha256")
    if not isinstance(digest, str) or not hmac.compare_digest(digest, expected_digest):
        return HandshakeVerdict(
            False, proto.Err.HASH_MISMATCH,
            f"dom_query.js digest {digest!r} != {expected_digest}",
        )

    token = hello.get("token")
    if not expected_token:
        # Fail closed: no stored credential means nothing may connect, rather
        # than everything.
        return HandshakeVerdict(
            False, proto.Err.UNAUTHORIZED,
            "no extension credential has been minted on this machine",
        )
    if not isinstance(token, str) or not hmac.compare_digest(token, expected_token):
        return HandshakeVerdict(False, proto.Err.UNAUTHORIZED, "bad token")

    # Last, so a would-be second client cannot use "occupied" as an oracle that
    # its token was otherwise correct... and equally, so a legitimate extension
    # reconnecting into an occupied slot is told the real reason.
    if occupied:
        return HandshakeVerdict(
            False, proto.Err.UNAUTHORIZED,
            "another extension is already connected",
        )

    return HandshakeVerdict(True)


# ─── The live connection ──────────────────────────────────────────────────


@dataclass
class LatchConnection:
    """One connected extension, and the calls in flight on it."""

    send_json: Callable[[dict], Any]
    browser_name: str = "other"
    protocol_version: int = proto.PROTOCOL_VERSION
    extension_version: str = ""
    connected_at: float = field(default_factory=time.monotonic)

    _next_id: int = 1
    _pending: dict[int, asyncio.Future] = field(default_factory=dict)
    _event_callbacks: list[Callable[[dict], None]] = field(default_factory=list)
    _closed: bool = False

    @property
    def connected(self) -> bool:
        return not self._closed

    def on_event(self, callback: Callable[[dict], None]) -> None:
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[[dict], None]) -> None:
        # A source that reconnects must be able to detach, or every reconnect
        # doubles the dispatch.
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    async def call(
        self, method: str, params: dict | None = None, *, timeout: float = _DEFAULT_CALL_TIMEOUT
    ) -> dict:
        """Send one request and await its reply.

        Raises `LatchCallError` on an error frame, `TimeoutError` on silence,
        and `LatchDisconnected` if the socket is gone. It never returns a
        sentinel: `.claude/rules/automation.md` records what happened when a
        failure came back as `"__FALLBACK__"` — a string that is not a failure
        report but an instruction to escalate a tier.

        The pending slot is released in `finally`, so a thousand timed-out calls
        leak nothing.
        """
        if self._closed:
            raise LatchDisconnected("the extension is not connected")

        call_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[call_id] = future

        try:
            await self.send_json({
                "type": proto.Frame.REQUEST,
                "id": call_id,
                "method": method,
                "params": params or {},
            })
            frame = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"{method} did not answer within {timeout}s"
            ) from e
        finally:
            self._pending.pop(call_id, None)

        if not frame.get("ok"):
            raise LatchCallError(
                frame.get("code", proto.Err.INTERNAL),
                str(frame.get("message", "")),
                method=method,
            )
        return frame.get("result") or {}

    def handle_frame(self, frame: dict) -> None:
        """Route one inbound frame. Never raises into the socket loop."""
        kind = frame.get("type")
        if kind == proto.Frame.RESPONSE:
            future = self._pending.get(frame.get("id"))
            if future is not None and not future.done():
                future.set_result(frame)
            return
        if kind == proto.Frame.EVENT:
            for callback in list(self._event_callbacks):
                try:
                    callback(frame)
                except Exception as e:
                    # One bad subscriber must not take the transport down.
                    logger.warning(f"[LATCH] event callback raised: {type(e).__name__}: {e}")
            return
        logger.debug(f"[LATCH] ignoring unexpected frame type {kind!r}")

    def close(self, reason: str = "closed") -> None:
        """Tear down, failing every call still waiting.

        Waiters are failed rather than left pending: a call that will never be
        answered must raise, not hang until its own timeout with no reason.
        """
        self._closed = True
        for call_id, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(LatchDisconnected(f"extension disconnected: {reason}"))
            self._pending.pop(call_id, None)
        self._event_callbacks.clear()


class LatchDisconnected(RuntimeError):
    """The extension is not connected, or went away mid-call."""


class LatchCallError(RuntimeError):
    """The extension answered with an error frame."""

    def __init__(self, code: int, message: str, *, method: str = "") -> None:
        super().__init__(f"{method or 'call'} failed [{code}]: {message}")
        self.code = code
        self.raw_message = message
        self.method = method


# ─── The registry ─────────────────────────────────────────────────────────
# One process, one extension. Module-level rather than app-state because the
# automation tier reaches for it and must not import the ASGI app to do so.

_connection: LatchConnection | None = None


def current_connection() -> LatchConnection | None:
    return _connection if (_connection is not None and _connection.connected) else None


def is_occupied() -> bool:
    return current_connection() is not None


def register(connection: LatchConnection) -> None:
    global _connection
    _connection = connection


def unregister(connection: LatchConnection, reason: str = "closed") -> None:
    global _connection
    connection.close(reason)
    if _connection is connection:
        _connection = None


@dataclass(frozen=True)
class LatchState:
    """What the affordance snapshot and the router ask about.

    Mirrors the shape the removed `cdp_state_snapshot()` had, because the same
    callers ask the same question: can the browser tier drive a real browser
    right now?
    """

    connected: bool
    browser_name: str = ""
    extension_version: str = ""
    protocol_version: int = 0


def latch_state_snapshot() -> LatchState:
    conn = current_connection()
    if conn is None:
        return LatchState(connected=False)
    return LatchState(
        connected=True,
        browser_name=conn.browser_name,
        extension_version=conn.extension_version,
        protocol_version=conn.protocol_version,
    )


def reset_state_for_test() -> None:
    """Test helper. Never call from production code."""
    global _connection
    if _connection is not None:
        _connection.close("test reset")
    _connection = None
