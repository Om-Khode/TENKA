"""Listener policy: what a transport permits, independent of what a device holds.

Milestone 6 exposes this daemon beyond loopback -- over two Tailscale tunnels
(`tailnet`, `funnel`). A third, a Cloudflare quick tunnel (`quick`), shipped in
6b and was removed in the same milestone: pairing was refused on it outright
(Cloudflare terminates TLS and would read both the pair code and the
`Set-Cookie` it redeems for), and with `allow_bearer=False` a cookie is the
*only* other way to authenticate -- but cookies are host-scoped, `quick`'s host
is `*.trycloudflare.com`, and no device's cookie (minted on loopback or over
the tailnet) is ever scoped to that host. So no device could ever authenticate
over it at all; every route it served demanded a capability nothing could
carry, and its actual, permanently reachable surface was the static landing
page -- which this project's public demo already serves better. See
`.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §5.5 for the full
chain, recorded there so the wall is not rediscovered by re-adding it.

`POST /v1/chat` reaches every intent, `code_executor` included, so
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

Milestone 6b adds `raisable`, a second explicit literal beside `ceiling`: what
a transport may EVER carry if a device's ceiling on it is deliberately, and
temporarily, lifted. `POLICIES` stays frozen module data -- a raise is a live
record kept elsewhere, scoped to one device and one policy, never a mutation
of the dict below. `effective()`'s third argument, `raised`, folds that record
in: `device_grants & (policy.ceiling | (policy.raisable & raised))`. A ceiling
still only narrows; `raisable` only says what narrowing is even reversible,
and only for the one transport where a raise was ever argued for.
"""
from __future__ import annotations

from dataclasses import dataclass

from .vault import Capability

# Every ceiling below is an explicit literal, and a test asserts that no
# enum-derived shorthand comes back.
#
# One used to stand here and feed `local`, `tailnet` and `funnel`, which meant
# a capability added to the enum flowed into the public listener for free.
# `EXECUTE` was exactly that capability: adding it would have handed a
# publicly reachable URL the right to run code on this machine without anyone
# deciding to. Milestone 6a.5 extended that discipline to all three, so the
# *next* capability is also granted nowhere by default and must be added by
# name after someone vets it for each transport.
#
# `local` is spelled out too, even though it does hold everything today: a
# derived ceiling there would keep re-introducing the same inheritance, and a
# reader comparing the three should be able to read them the same way.


@dataclass(frozen=True)
class ListenerPolicy:
    """What one kind of listening socket is trusted to carry, regardless of
    which device's grants show up on it."""

    name: str                        # "local" | "tailnet" | "funnel"
    admin: bool                      # may reach device/transport management
    allow_bearer: bool               # may authenticate with an Authorization header
    secure_cookie: bool              # sets the Secure flag
    ceiling: frozenset[Capability]
    raisable: frozenset[Capability]  # what this transport may EVER carry under a raise
    # May a device credential ever be minted over this transport at all --
    # `pairing_denied_by_transport()` in `routes/pairing.py` is `not pairable`.
    # An explicit literal on every policy, like `ceiling` and `raisable`, and
    # for the identical reason: the removed `quick` policy used to be singled
    # out by `policy.name == "quick"`, a hardcoded name check that is exactly
    # what this project's standing rule forbids (behaviour is data). With
    # `quick` gone the check would have degenerated into dead code that always
    # answered `False` -- so `pairable` earns its place as a field precisely
    # because a *future* transport must declare, explicitly, whether a
    # credential may be minted over it, rather than silently inheriting "yes"
    # by omission.
    pairable: bool


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
        # All seven, named. The operator at the keyboard keeps full power --
        # 6a.5 is not a downgrade of the local path -- but naming them is what
        # makes the *other* three ceilings meaningful: a new capability has to
        # be added in four places, and three of those are a decision.
        ceiling=frozenset({
            Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
            Capability.SCREEN, Capability.FILES, Capability.SYSTEM_CONTROL,
            Capability.EXECUTE,
        }),
        # Already holds everything; there is nothing left for a raise to add.
        raisable=frozenset(),
        pairable=True,
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
        # Everything except EXECUTE and SYSTEM_CONTROL. WireGuard protects the
        # bytes, not the endpoint: a phone on the tailnet is a phone somebody
        # can pick up, and "read her transcripts" and "run code on her
        # machine" are not the same trust. SYSTEM_CONTROL is dropped for the
        # same reason -- PATCH /v1/settings turns the camera on and speaker
        # verification off, which is a change to the machine, not a request
        # of it.
        ceiling=frozenset({
            Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
            Capability.SCREEN, Capability.FILES,
        }),
        # The only transport raisable at all, and the reason is a second
        # credential, not the tunnel's encryption. Reaching this listener
        # already required a Tailscale login before a TENKA device
        # credential was even presented -- an independent gate a leaked pair
        # code cannot pass on its own. That is what makes lifting EXECUTE and
        # SYSTEM_CONTROL here, deliberately and temporarily, a decision about
        # a vetted machine on the operator's own tailnet, not a hole opened
        # in a listener anyone with a URL can reach.
        raisable=frozenset({Capability.EXECUTE, Capability.SYSTEM_CONTROL}),
        pairable=True,
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
        # Same set as tailnet, and the same two omissions, for a stronger
        # reason: this URL is reachable by anyone who has it. CHAT_SEND alone
        # reaches every intent through POST /v1/chat, so without the EXECUTE
        # split a leaked pair code was a remote shell. The ceiling is what
        # makes it only a conversation.
        ceiling=frozenset({
            Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
            Capability.SCREEN, Capability.FILES,
        }),
        # Never raisable: this URL is publicly reachable by anyone who holds
        # it, with no second credential gating the connection the way a
        # tailnet login gates `tailnet`. There is no vetted machine on the
        # other end to decide to trust further -- only "whoever has the
        # link" -- so there is nothing here a raise could safely lift.
        raisable=frozenset(),
        pairable=True,
    ),
    # The Cloudflare quick tunnel that used to stand here (`quick`) was
    # removed in Milestone 6b, not narrowed: no device could ever
    # authenticate over it. Pairing was refused on it outright (spec §5.5--
    # Cloudflare terminates TLS and would read both the pair code and the
    # `Set-Cookie` it redeems for), and `allow_bearer=False` meant the cookie
    # was the only other door -- but cookies are host-scoped, `quick`'s host
    # was `*.trycloudflare.com`, and no device paired on loopback or over the
    # tailnet ever holds a cookie scoped to that host. So its `OBSERVE`
    # ceiling was decoration: every route it served demanded a capability
    # nothing could ever present, and its one permanently reachable surface
    # was the static landing page, which this project's public demo already
    # serves better. `funnel` covers the audience it existed for -- a public
    # HTTPS URL needing nothing installed on the phone. See
    # `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §5.5.
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


def effective(
    device_grants: frozenset[Capability],
    policy: ListenerPolicy,
    raised: frozenset[Capability] = frozenset(),
) -> frozenset[Capability]:
    """What a device may actually do on this listener: the intersection of
    what it was issued and what this transport is willing to carry.

    Never a union over `device_grants` -- a policy can only take capabilities
    away from a device, never add ones the device was never granted. `raised`
    is the one thing that can widen the *transport* side of that intersection,
    and only within `raisable`, which is itself a fixed, per-policy ceiling on
    what a raise could ever reach. `raised` defaults to empty, so calling this
    with two arguments -- every call site before Milestone 6b -- is identical
    to 6a.5's behaviour: no raise, no change.
    """
    return device_grants & (policy.ceiling | (policy.raisable & raised))
