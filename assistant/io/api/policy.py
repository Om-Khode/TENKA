"""Listener policy: what a transport permits, independent of what a device holds.

Milestone 6 exposes this daemon beyond loopback -- over a Tailscale tunnel, and
over a Cloudflare quick tunnel where Cloudflare terminates TLS and can read the
plaintext. `POST /v1/chat` reaches every intent, `code_executor` included, so
the API can run arbitrary code on this machine. A device's grants (`vault.py`)
say what it was issued; a `ListenerPolicy` says what the *transport it arrived
on* is trusted to carry at all. The two are combined with `effective()`, an
intersection that can only narrow, never widen, what a device may do.

Policy is keyed on the local port the connection was accepted on -- read from
the ASGI scope's own server address, which the client cannot influence -- and
not on anything the client sends. Two alternatives were considered and
rejected:

- The client address. `cloudflared` and `tailscale funnel` both connect to the
  daemon from 127.0.0.1, so tunnelled traffic arrives with a loopback source
  address indistinguishable from a truly local caller. An "is the peer local?"
  check would hand every tunnel full admin rights.
- The `Host` header. That is attacker-controlled input; letting it select a
  policy is privilege escalation by request field.

A listening socket's own address cannot be forged by a client. Binding policy
to the port the daemon chose to listen on, rather than to anything carried in
the request, is what makes this fail closed instead of fail open.
"""
from __future__ import annotations

from dataclasses import dataclass

from .vault import Capability

# `local`, `tailnet`, and `funnel` derive their ceiling from the enum itself,
# so a capability added to `Capability` later flows into all three
# automatically -- appropriate for transports already trusted with
# everything. `quick` spells its ceiling out as an explicit literal instead
# (see below) precisely so that inheritance does NOT happen there: a
# disclosure-limited transport must never silently gain a capability nobody
# vetted for it just because the enum grew.
_ALL_CAPABILITIES = frozenset(Capability)


@dataclass(frozen=True)
class ListenerPolicy:
    """What one kind of listening socket is trusted to carry, regardless of
    which device's grants show up on it."""

    name: str                        # "local" | "tailnet" | "funnel" | "quick"
    admin: bool                      # may reach device/transport management
    allow_bearer: bool               # may authenticate with an Authorization header
    secure_cookie: bool              # sets the Secure flag
    ceiling: frozenset[Capability]


POLICIES: dict[str, ListenerPolicy] = {
    # Loopback only. The one listener a caller cannot reach without already
    # having a foothold on this machine, so it alone may manage devices and
    # accept a bearer token (a header, unlike a cookie, survives a copy-paste
    # into a script -- fine on loopback, not fine anywhere a network sees it).
    "local": ListenerPolicy(
        name="local",
        admin=True,
        allow_bearer=True,
        secure_cookie=False,   # plain http on loopback; there is no TLS to require
        ceiling=_ALL_CAPABILITIES,
    ),
    # Tailscale tunnel: WireGuard end-to-end, the operator's own tailnet.
    # Trusted with every capability a device might hold, but never with admin
    # (device/transport management stays loopback-only) and never with a
    # bearer header (cookies only, once traffic leaves the machine).
    "tailnet": ListenerPolicy(
        name="tailnet",
        admin=False,
        allow_bearer=False,
        secure_cookie=True,
        ceiling=_ALL_CAPABILITIES,
    ),
    # `tailscale funnel` -- a top-level command, not a `serve` flag; there is
    # no `tailscale serve --funnel`. Publicly reachable from the open
    # internet by anyone with the URL, but TLS is still terminated on this
    # machine by the Tailscale client, not decrypted at a relay in between --
    # Tailscale's infrastructure never sees the plaintext. Same ceiling as
    # tailnet for that reason: the plaintext exposure is identical, only the
    # audience is wider. Public reachability is exactly why admin and bearer
    # stay off.
    "funnel": ListenerPolicy(
        name="funnel",
        admin=False,
        allow_bearer=False,
        secure_cookie=True,
        ceiling=_ALL_CAPABILITIES,
    ),
    # Cloudflare quick tunnel: Cloudflare terminates TLS and can read the
    # plaintext. The ceiling is OBSERVE alone -- watching her work. Even a
    # device issued every capability is limited to her status, her telemetry,
    # the live event stream and how she is configured; never to acting
    # (CHAT_SEND, FILES, SYSTEM_CONTROL), and never to what she has stored.
    #
    # SCREEN and RECALL are excluded for the same reason as each other, and a
    # different one from the acting grants: this isn't about what an attacker
    # could *do*, it's about what Cloudflare could *see*. Screen capture is
    # the highest-bandwidth disclosure in the API -- but RECALL is the widest.
    # It carries the entire knowledge graph and every transcript, and since
    # `read_screen` and `camera_look` are intents, her narration of what was
    # on screen is *in* those transcripts. Excluding SCREEN while admitting
    # stored data withheld the photograph and shipped the description of it.
    # Reading stored data over this transport joins acting behind a
    # deliberate, expiring, audited raise.
    "quick": ListenerPolicy(
        name="quick",
        admin=False,
        allow_bearer=False,
        secure_cookie=True,
        # Deliberately an explicit literal, not `_ALL_CAPABILITIES - {...}`.
        # Spelling out what IS allowed, rather than what is excluded, means a
        # future capability added to the enum is granted nowhere by default
        # and must be added here by name after someone vets it for the one
        # transport a third party can read. Rewriting this as a subtraction
        # would look like a tidy-up but inverts the safety property: the
        # next new capability would then be granted automatically over
        # exactly the listener that must never get one for free.
        ceiling=frozenset({Capability.OBSERVE}),
    ),
}


def policy_for_port(port: int, registry: dict[int, str]) -> ListenerPolicy | None:
    """Look up the policy for the port a connection was accepted on.

    Fails closed twice over: an unregistered port grants nothing (nobody
    declared what that socket is), and a registry entry naming a policy that
    does not exist in `POLICIES` grants nothing either, rather than falling
    back to some default. A typo in the registry must lose capabilities, not
    silently keep whatever was there before.
    """
    name = registry.get(port)
    if name is None:
        return None
    return POLICIES.get(name)


def effective(device_grants: frozenset[Capability], policy: ListenerPolicy) -> frozenset[Capability]:
    """What a device may actually do on this listener: the intersection of
    what it was issued and what this transport is willing to carry.

    Never a union -- a policy's ceiling can only take capabilities away from
    a device, never add ones the device was never granted.
    """
    return device_grants & policy.ceiling
