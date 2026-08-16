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
machine (`tailscale --version` -> `1.102.2`), via `tailscale serve --help`
and `tailscale funnel --help` on 2026-08-16:

- `tailscale serve <target>` -- `--bg` backgrounds (daemonises) the command;
  `--https value` selects the public HTTPS port (443 is the documented
  default). `serve --help` reads verbatim: "To share a local server on the
  internet, use `tailscale funnel`" -- confirming there is no `--funnel` flag
  on `serve`.
- `tailscale funnel <target>` -- its own top-level command (`USAGE
  tailscale funnel <target>`), not a flag on `serve`. Same `--bg` and
  `--https` flags exist on `funnel` too, but the documented example
  (`tailscale funnel --bg 3000`) passes the local port bare, which is the
  form used below.
- Un-serving: Tailscale's own docs (and the `serve`/`funnel status --json`
  shape) confirm turning a mapping off means re-issuing the same identifying
  flags with `off` appended, not just killing the process that requested the
  mapping -- `--bg` daemonises and the invoking process exits on its own, so
  killing it again touches nothing. Both adapters' `stop_command` therefore
  return an argv rather than `None`.

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
        """`tailscale serve --bg --https 443 http://127.0.0.1:{port}` --
        verified against `tailscale serve --help` on the installed 1.102.2
        binary (see module docstring). Built from the integer *port* and
        module constants only; a non-numeric *port* raises rather than
        reaching the argv (spec §8's subprocess-injection row)."""
        port = int(port)
        return [
            "tailscale", "serve", "--bg", "--https", "443",
            f"http://127.0.0.1:{port}",
        ]

    def stop_command(self, port: int) -> list[str] | None:
        """`tailscale serve --https 443 off` -- `--bg` daemonises and the
        spawning process exits on its own, so only the explicit `off` form
        un-serves it. Identified by the public `--https` port, the same
        flag `command()` used to create the mapping; *port* (the local
        target) plays no part in which mapping `off` removes."""
        return ["tailscale", "serve", "--https", "443", "off"]


# ─── Funnel adapter ───────────────────────────────────────────────────────────

class FunnelAdapter(_TailscaleAdapterBase):
    """`tailscale funnel` -- the same machine and the same locally-terminated
    TLS as `tailnet`, but published to the open internet. One credential
    (the URL) instead of two. Never raisable (spec §3: `raisable=frozenset()`
    in `policy.py`)."""

    name = "funnel"

    def command(self, port: int) -> list[str]:
        """`tailscale funnel --bg {port}` -- verified against `tailscale
        funnel --help` on the installed 1.102.2 binary: `funnel` is its own
        top-level command (`USAGE  tailscale funnel <target>`), not a flag
        on `serve` -- there is no `tailscale serve --funnel`. Built from the
        integer *port* and module constants only, exactly like
        `TailnetAdapter.command`."""
        port = int(port)
        return ["tailscale", "funnel", "--bg", str(port)]

    def stop_command(self, port: int) -> list[str] | None:
        """`tailscale funnel {port} off` -- the target argument that
        identified the mapping, with `off` appended; `--bg` daemonises so,
        as with `tailnet`, only this explicit form un-serves it."""
        return ["tailscale", "funnel", str(int(port)), "off"]


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import transport_registry
    transport_registry.register("tailnet", TailnetAdapter())
    transport_registry.register("funnel", FunnelAdapter())

_register()
