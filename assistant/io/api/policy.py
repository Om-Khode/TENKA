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
# deciding to. `quick` already spelled its ceiling out for this reason (the
# comment on it argues the point); Milestone 6a.5 extends that discipline to
# all four, so the *next* capability is also granted nowhere by default and
# must be added by name after someone vets it for each transport.
#
# `local` is spelled out too, even though it does hold everything today: a
# derived ceiling there would keep re-introducing the same inheritance, and a
# reader comparing the four should be able to read them the same way.


@dataclass(frozen=True)
class ListenerPolicy:
    """What one kind of listening socket is trusted to carry, regardless of
    which device's grants show up on it."""

    name: str                        # "local" | "tailnet" | "funnel" | "quick"
    admin: bool                      # may reach device/transport management
    allow_bearer: bool               # may authenticate with an Authorization header
    secure_cookie: bool              # sets the Secure flag
    ceiling: frozenset[Capability]
    raisable: frozenset[Capability]  # what this transport may EVER carry under a raise


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
        # Deliberately an explicit literal, not "everything minus {...}".
        # Spelling out what IS allowed, rather than what is excluded, means a
        # future capability added to the enum is granted nowhere by default
        # and must be added here by name after someone vets it for the one
        # transport a third party can read. Rewriting this as a subtraction
        # would look like a tidy-up but inverts the safety property: the
        # next new capability would then be granted automatically over
        # exactly the listener that must never get one for free.
        ceiling=frozenset({Capability.OBSERVE}),
        # Never raisable: Cloudflare terminates TLS and reads the plaintext
        # of everything this listener carries. A raise widens what a raised
        # device may do, not who else can read it -- and on this transport a
        # third party reads it regardless, so there is no capability whose
        # exposure a raise could make acceptable.
        raisable=frozenset(),
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
