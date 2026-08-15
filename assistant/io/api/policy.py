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

- The client address. `cloudflared` and `tailscale serve` both connect to the
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

from assistant.io.api.vault import Capability

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
    # `tailscale serve --funnel`: still Tailscale-terminated TLS, reachable
    # from the public internet by anyone with the URL. Same ceiling as
    # tailnet -- the transport is still Tailscale's, not a relay's -- but
    # public reachability is exactly why admin and bearer stay off.
    "funnel": ListenerPolicy(
        name="funnel",
        admin=False,
        allow_bearer=False,
        secure_cookie=True,
        ceiling=_ALL_CAPABILITIES,
    ),
    # Cloudflare quick tunnel: Cloudflare terminates TLS and can read the
    # plaintext. The ceiling is read-only -- CHAT alone -- so even a device
    # issued every capability is limited to reading history and status
    # through this transport, never to acting (CHAT_SEND, FILES,
    # SYSTEM_CONTROL). SCREEN is excluded too, but for a different reason
    # than the others: this isn't about what an attacker could *do*, it's
    # about what Cloudflare could *see*. Screen capture is the
    # highest-bandwidth disclosure in the API, and this is the one listener
    # a third party's infrastructure can observe.
    "quick": ListenerPolicy(
        name="quick",
        admin=False,
        allow_bearer=False,
        secure_cookie=True,
        ceiling=frozenset({Capability.CHAT}),
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
