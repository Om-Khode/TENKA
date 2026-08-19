"""The listener port map -- one fixed port per policy, and KI-17.

Milestone 6 binds one `uvicorn.Server` per listener (`local`, `tailnet`,
`funnel`), all serving the same ASGI app, separated by the port each
socket was bound to (`policy.py` argues why port, not host or peer address, is
the only thing a client cannot forge). This module is the one place that
port map is declared.

A fourth listener, `quick` (a Cloudflare tunnel), held offset 3 here through
Milestone 6b and was removed in the same milestone -- no device could ever
authenticate over it (`policy.py`'s module docstring has the full argument).

KI-17: `policy_for_port` keys on the accepting port, which is correct and
unforgeable -- but if a tunnel is pointed at the *existing* Studio port,
every tunnelled request resolves to `POLICIES["local"]`: admin, bearer, and a
ceiling holding `EXECUTE` and `SYSTEM_CONTROL`. Every ceiling 6a.5 built is
bypassed, not by a bug but by the obvious implementation. That is why this
map exists as its own module rather than as a literal scattered across the
transport adapters: one place to get it right, one place to audit.

Ports are fixed offsets from `config.STUDIO_API_PORT`, not kernel-assigned
(`bind(port 0)`). Kernel assignment would make KI-17 unreachable by
construction -- a tunnel could never be pointed at a port that changes every
run -- but the operator's `tailscale serve --bg https / http://127.0.0.1:8788`
configuration has to survive a reboot, and an ephemeral port invalidates that
config on every restart. Fixed ports were chosen anyway, and three other
defence layers replace what ephemeral ports would have bought:

1. TENKA builds every tunnel's argv itself; the target port comes from this
   registry, never from an operator's keyboard. A port typed by a human is a
   port a human can type wrong, and typing 8787 is KI-17. A spawn-time
   assertion checks the real argument vector against `port_for()`.
2. A preflight reconciliation, on every start, reads
   `tailscale serve status --json` and refuses to start when a pre-existing
   mapping targets any port other than that transport's own -- catching a
   `tailscale serve` config the operator hand-set months ago and forgot.
3. The local listener's `HostGate` accepts loopback `Host` names only
   (`127.0.0.1`, `localhost`, `[::1]`). A tunnel forwards the public
   authority in `Host` (`*.ts.net`, `*.trycloudflare.com`), so a tunnelled
   request arriving on the local port is refused with 421 before
   authentication, before policy lookup, before any route runs. This is the
   load-bearing layer: it holds even against a tunnel TENKA never launched
   and knows nothing about, which is more than an ephemeral port could ever
   offer.

None of the three is implemented here -- this module only records why fixed
ports need them.
"""
from __future__ import annotations

# ─── Port map ────────────────────────────────────────────────────────────

# Offsets from `config.STUDIO_API_PORT`, one per listener policy. Declared as
# a literal dict, not derived from `POLICIES` or the `Capability` enum: a
# policy added to `policy.py` without a corresponding entry here is a
# listener nobody gave a port, and must fail loudly rather than inherit one.
LISTENER_OFFSETS: dict[str, int] = {
    "local": 0,
    "tailnet": 1,
    "funnel": 2,
}


# ─── Lookups ─────────────────────────────────────────────────────────────

def port_for(policy_name: str, base_port: int) -> int:
    """The fixed port for one listener policy.

    Raises `KeyError` on an unknown policy name. There is no fallback port
    and no default -- a policy nobody mapped must fail closed, not silently
    borrow the port of a different listener.
    """
    return base_port + LISTENER_OFFSETS[policy_name]


def policy_name_for_port(port: int, base_port: int) -> str | None:
    """The policy name a port maps to, or `None` if no listener claims it.

    `None` -- not a default policy -- is deliberate: an unregistered port
    must resolve to nothing, the same way `policy_for_port` in `policy.py`
    grants nothing for a port absent from the registry.
    """
    offset = port - base_port
    for name, listener_offset in LISTENER_OFFSETS.items():
        if listener_offset == offset:
            return name
    return None


def local_port(base_port: int) -> int:
    """The port `local` holds -- the one no other listener may ever share.

    A thin convenience over `port_for("local", base_port)`, kept as its own
    function because KI-17 checks ("is this the local port?") are a distinct
    question from "what port does policy X hold?", and read better spelled
    out at call sites that only care about the former.
    """
    return port_for("local", base_port)
