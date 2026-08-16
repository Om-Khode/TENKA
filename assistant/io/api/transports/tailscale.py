# assistant/io/api/transports/tailscale.py
"""Tailscale transport adapters -- `tailnet` (`tailscale serve`) and `funnel`
(`tailscale funnel`).

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.3 (L1,
L2), §4, §8.

Both share the same TLS-on-this-machine story (Tailscale terminates it, the
tunnel never hands plaintext to a third party) and the same `*.ts.net`
hostname shape; they differ only in reach -- `tailnet` is reachable by
devices signed into the operator's own tailnet, `funnel` publishes the same
socket to the open internet -- and therefore in `command()` and their
`POLICIES` name. `_TailscaleAdapterBase` holds what is shared: hostname
recognition and the KI-17 layer-2 preflight.

**Verified command forms**, against the binary actually installed on this
machine (`tailscale --version` -> `1.102.2`), via `tailscale serve --help`,
`tailscale funnel --help`, `tailscale serve status --json` and
`tailscale funnel status --json` on 2026-08-16 (all read-only -- none of
them create, change or reset a mapping):

- `tailscale serve <target>` -- `--bg` backgrounds (daemonises) the command;
  `--https value` selects the public HTTPS port. `serve --help` reads
  verbatim: "To share a local server on the internet, use `tailscale
  funnel`" -- confirming there is no `--funnel` flag on `serve`.
- `tailscale funnel <target>` -- its own top-level command (`USAGE
  tailscale funnel <target>`), not a flag on `serve`. Same `--bg` and
  `--https` flags exist on `funnel` too.
- **Public port split (fix round 1).** `--https value` defaults to 443 for
  *both* verbs, and Tailscale keys a serve/funnel mapping on the public
  port, not on which local target it forwards to. The first draft of this
  module let both adapters default to 443, which would have made starting
  `funnel` silently overwrite the `tailnet` mapping (or vice versa) --
  running both simultaneously, which this milestone requires, is exactly
  the scenario that collides. Both adapters now pass `--https` explicitly
  and to *different* ports: `tailnet` takes `8443`, `funnel` takes `443`.
  Funnel is restricted by Tailscale itself to ports 443, 8443 or 10000
  (confirmed against Tailscale's own Funnel docs,
  https://tailscale.com/kb/1223/funnel: "Funnel can only listen on ports
  443, 8443, and 10000"); 443 is assigned to it because its URL is the one
  that might be typed or pasted by hand and needs no port suffix, while
  `tailnet`'s URL is always generated and copied, never typed.
  `test_the_two_transports_never_share_a_public_port` pins the two apart so
  a future edit cannot collapse them back onto one.
- Un-serving. `tailscale serve --help`'s own `SUBCOMMANDS` list (`status`,
  `reset`, `drain`, `clear`, `advertise`, `get-config`, `set-config`) has no
  `off` entry, and the `<target>` grammar it documents ("a file, directory,
  text, or ... a service") means an unqualified `off` risks being parsed as
  literal text to serve rather than a request to stop serving. `off` is not
  a guess, though: Tailscale's own current CLI reference pages document it
  explicitly and give this exact shape --
  https://tailscale.com/docs/reference/tailscale-cli/serve: "To turn off a
  `tailscale serve` command, you can add `off` to the end of the command
  you used to turn it on... You can omit the `<target>` argument, so these
  2 commands are equivalent" -- and the identical wording appears on
  https://tailscale.com/docs/reference/tailscale-cli/funnel for `funnel`.
  It is absent from `--help`'s `SUBCOMMANDS` because it is not a subcommand
  -- it is a special form of the primary `<target>` grammar, the same way a
  bare port number or a URL is a `<target>` without appearing in that list.
  `reset` was considered and rejected: it wipes the *entire* serve config,
  including any mapping the operator set up by hand for something
  unrelated -- the KI-17 hazard pointed the other way, clobbering
  configuration this adapter does not own. `off`, re-issuing the same
  `--https` flag `command()` used, is the targeted alternative.
  This was **not** run on this machine to confirm empirically -- doing so
  would require an active mapping to toggle off, which the read-only
  constraint on this fix round rules out. Because of that, `stop_command`
  cannot be trusted on documentation alone: its docstring below states the
  verification obligation explicitly, and it is on the caller
  (`TransportManager`, Task 9) to discharge it by re-reading `... status
  --json` after running this argv and confirming the mapping is actually
  gone before treating the stop as successful.
- Both adapters' `stop_command` return an argv rather than `None`: `--bg`
  daemonises and the invoking process exits on its own, so killing it again
  touches nothing; only the `off` argv un-serves it.
- Both `command()` forms are now the same shape (verb, `--bg`, `--https`,
  public port, local target URL) differing only in the verb and the public
  port -- the earlier asymmetry (a full URL for `tailnet`, a bare port for
  `funnel`) was gratuitous once both carry an explicit public port.

Layering: `io/api/` may import `core/` and `config` only. This module imports
neither -- `subprocess`, `json`, `logging`, `re` and `urllib.parse` are
stdlib -- and reaches upward at nothing.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Tailscale keys a serve/funnel mapping on the *public* port (`--https`),
# not on the local target it forwards to -- so `tailnet` and `funnel` must
# never share one, or starting the second silently overwrites the first's
# mapping (fix round 1, F2). Funnel may only ever use 443, 8443 or 10000
# (https://tailscale.com/kb/1223/funnel); 443 goes to `funnel` because its
# URL is the one an operator might type or paste by hand and needs no port
# suffix, `tailnet`'s URL being always generated rather than typed.
_TAILNET_PUBLIC_PORT = 8443
_FUNNEL_PUBLIC_PORT = 443

# `tailscale serve status --json` is a local, already-running daemon query --
# fast in practice. The timeout exists so a hung `tailscaled` cannot hang a
# transport start indefinitely; a timeout degrades to a warning exactly like
# any other unparseable output (see `_run_serve_status`).
_PREFLIGHT_TIMEOUT_SECONDS = 10.0

# A `*.ts.net` MagicDNS name: one or more dot-separated labels (letters,
# digits, internal hyphens) ending in the literal suffix `ts.net`. Anchored
# full-match so "laptop.ts.net.evil.com" (suffix confusion) and
# "evil.example.com" (wrong domain entirely) both fail -- a name announced by
# the tunnel subprocess becomes a trusted `Host` and `Origin` (spec §8), so
# this must reject anything outside the provider's own shape rather than
# accept anything that merely looks like a hostname.
_TS_NET_HOSTNAME_RE = re.compile(
    r"^(?:[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+ts\.net$"
)

# The first `https://...` substring on a line of tunnel-process stdout, e.g.
# "Available within your tailnet: https://laptop.tail1234.ts.net/".
_URL_IN_LINE_RE = re.compile(r"https://\S+")


# ─── Preflight parsing (shared, KI-17 layer 2) ───────────────────────────────

def parse_serve_status(payload: dict, expected_port: int) -> str | None:
    """Reconcile `tailscale serve status --json` with *expected_port*.

    Returns `None` when every existing `Web` mapping's proxy target already
    points at *expected_port* (including the common case of no mappings at
    all -- confirmed against this machine's real, currently-clear
    `tailscale serve status --json`, which prints a bare `{}`). Returns a
    refusal sentence naming the offending port -- never a hostname, a token
    or a path -- when a mapping targets some other port: the KI-17 scenario
    of a `tailscale serve` configuration the operator hand-set months ago,
    still pointed at the pre-6b Studio port.

    Degrades to a logged warning and `None` -- not a raised exception -- on a
    shape this cannot recognise. Layer 3 (the per-listener `Host` gate)
    contains the failure either way; a preflight that hard-fails on an
    unrecognised Tailscale version would take the whole transport down for a
    formatting change.
    """
    try:
        web = payload.get("Web") or {}
        offending_ports: set[int] = set()
        for mapping in web.values():
            handlers = mapping.get("Handlers") or {}
            for handler in handlers.values():
                proxy = handler.get("Proxy")
                if not proxy:
                    continue
                port = urlparse(proxy).port
                if port is not None and port != expected_port:
                    offending_ports.add(port)
    except (AttributeError, TypeError) as exc:
        logger.warning(
            "tailscale serve status --json produced a shape preflight does "
            "not recognise (%s); degrading to a warning rather than "
            "blocking the transport from starting",
            exc,
        )
        return None

    if not offending_ports:
        return None
    ports = ", ".join(str(p) for p in sorted(offending_ports))
    return (
        f"tailscale serve already has a mapping pointed at port {ports}, "
        f"not this transport's own port {expected_port} -- run "
        f"'tailscale serve reset' or correct the mapping before starting "
        f"this transport"
    )


def _run_serve_status() -> dict | None:
    """Run `tailscale serve status --json` and parse it, or `None` on any
    failure to run or parse -- a missing binary, a timeout, or output that
    is not valid JSON all degrade the same way as an unrecognised shape."""
    try:
        result = subprocess.run(
            ["tailscale", "serve", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning(
            "could not read 'tailscale serve status --json' (%s); "
            "degrading to a warning rather than blocking the transport "
            "from starting",
            exc,
        )
        return None


# ─── Shared base ──────────────────────────────────────────────────────────────

class _TailscaleAdapterBase:
    """Hostname recognition and the KI-17 layer-2 preflight, shared by both
    Tailscale adapters. `command()`, `name` and `stop_command()` differ per
    subclass and are declared there."""

    def hostname_from(self, line: str) -> str | None:
        match = _URL_IN_LINE_RE.search(line)
        if match is None:
            return None
        host = urlparse(match.group(0)).hostname
        if host is None or not _TS_NET_HOSTNAME_RE.fullmatch(host):
            return None
        return host

    def preflight(self, port: int) -> str | None:
        payload = _run_serve_status()
        if payload is None:
            return None
        return parse_serve_status(payload, port)


# ─── Tailnet adapter ──────────────────────────────────────────────────────────

class TailnetAdapter(_TailscaleAdapterBase):
    """`tailscale serve` -- reachable only by devices signed into the
    operator's own tailnet. The only transport in the system whose ceiling
    may ever be raised (spec §3), because reaching it already required a
    Tailscale login before a TENKA device credential was even presented."""

    name = "tailnet"

    def command(self, port: int) -> list[str]:
        """`tailscale serve --bg --https 8443 http://127.0.0.1:{port}` --
        verified against `tailscale serve --help` on the installed 1.102.2
        binary (see module docstring). Public port `8443` (module constant
        `_TAILNET_PUBLIC_PORT`), never `funnel`'s `443` -- Tailscale keys a
        mapping on the public port, so sharing one with `funnel` would let
        starting either overwrite the other's mapping while both transports
        must run at once. Built from the integer *port* and module
        constants only; a non-numeric *port* raises rather than reaching
        the argv (spec §8's subprocess-injection row)."""
        port = int(port)
        return [
            "tailscale", "serve", "--bg", "--https", str(_TAILNET_PUBLIC_PORT),
            f"http://127.0.0.1:{port}",
        ]

    def stop_command(self, port: int) -> list[str] | None:
        """`tailscale serve --https 8443 off` -- re-issues the same
        `--https` flag `command()` used to create the mapping, with `off`
        appended, per Tailscale's own documented form (module docstring);
        *port* (the local target) plays no part in which mapping `off`
        removes. `--bg` daemonises and the spawning process exits on its
        own, so only this explicit form un-serves it.

        **Caller obligation (unverified on documentation alone -- fix round
        1, F1):** this argv was not exercised against a live mapping on this
        machine, because doing so would require creating one, which this
        fix round's read-only constraint ruled out. The caller
        (`TransportManager`, Task 9) MUST run this, then re-read `tailscale
        serve status --json` and confirm no `Web` mapping still proxies to
        this transport's own port, before treating the stop as successful.
        If the mapping is still present, the stop must fail loudly -- an
        internet-facing listener that silently stayed up is the worst
        failure mode in this milestone, worse than a stop that visibly
        failed."""
        return ["tailscale", "serve", "--https", str(_TAILNET_PUBLIC_PORT), "off"]


# ─── Funnel adapter ───────────────────────────────────────────────────────────

class FunnelAdapter(_TailscaleAdapterBase):
    """`tailscale funnel` -- the same machine and the same locally-terminated
    TLS as `tailnet`, but published to the open internet. One credential
    (the URL) instead of two. Never raisable (spec §3: `raisable=frozenset()`
    in `policy.py`)."""

    name = "funnel"

    def command(self, port: int) -> list[str]:
        """`tailscale funnel --bg --https 443 http://127.0.0.1:{port}` --
        verified against `tailscale funnel --help` on the installed 1.102.2
        binary: `funnel` is its own top-level command (`USAGE  tailscale
        funnel <target>`), not a flag on `serve` -- there is no `tailscale
        serve --funnel`. Public port `443` (module constant
        `_FUNNEL_PUBLIC_PORT`), never `tailnet`'s `8443`, for the same
        mapping-collision reason documented on `TailnetAdapter.command`; 443
        is one of the three ports Tailscale Funnel is restricted to
        (443/8443/10000) and is the one assigned here because a funnel URL
        may be typed or pasted by hand and needs no port suffix. Same shape
        as `TailnetAdapter.command` (verb, `--bg`, `--https`, public port,
        local target URL) -- the earlier asymmetry (a bare port instead of a
        URL) was gratuitous once both carry an explicit public port. Built
        from the integer *port* and module constants only; a non-numeric
        *port* raises rather than reaching the argv (spec §8's
        subprocess-injection row)."""
        port = int(port)
        return [
            "tailscale", "funnel", "--bg", "--https", str(_FUNNEL_PUBLIC_PORT),
            f"http://127.0.0.1:{port}",
        ]

    def stop_command(self, port: int) -> list[str] | None:
        """`tailscale funnel --https 443 off` -- re-issues the same
        `--https` flag `command()` used, with `off` appended, mirroring
        `TailnetAdapter.stop_command`; *port* plays no part in which mapping
        `off` removes. `--bg` daemonises so, as with `tailnet`, only this
        explicit form un-serves it.

        **Caller obligation -- identical to `TailnetAdapter.stop_command`,
        see its docstring:** not exercised against a live mapping on this
        machine (fix round 1, F1); `TransportManager` (Task 9) MUST verify
        via `tailscale funnel status --json` that the mapping is actually
        gone before treating the stop as successful, and fail loudly if it
        is not."""
        return ["tailscale", "funnel", "--https", str(_FUNNEL_PUBLIC_PORT), "off"]


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import transport_registry
    transport_registry.register("tailnet", TailnetAdapter())
    transport_registry.register("funnel", FunnelAdapter())

_register()
