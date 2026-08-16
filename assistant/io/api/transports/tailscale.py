# assistant/io/api/transports/tailscale.py
"""Tailscale transport adapters -- `tailnet` (`tailscale serve`) and `funnel`
(`tailscale funnel`).

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.3 (L1,
L2), §4, §8.

Both share the same TLS-on-this-machine story (Tailscale terminates it, the
tunnel never hands plaintext to a third party) and the same `*.ts.net`
hostname shape; they differ only in reach -- `tailnet` is reachable by
devices signed into the operator's own tailnet, `funnel` publishes the same
socket to the open internet -- and therefore in `command()`, their status
verb, their public port, and their `POLICIES` name. `_TailscaleAdapterBase`
holds what is shared: hostname recognition and the KI-17 layer-2 preflight.

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
  `test_the_two_transports_never_share_a_public_port` pins the two apart
  *and* pins which adapter gets which port, so a future edit cannot
  collapse them back onto one port or silently swap the assignment.
- **`Web` is one shared document (fix round 2, Important 1).** `tailscale
  serve status --json` and `tailscale funnel status --json` describe the
  *same* underlying per-node serve configuration -- a funnel mapping is a
  serve mapping with `AllowFunnel` set, which is exactly why the public-port
  split above matters: both transports' mappings live in one `Web` dict,
  keyed by `"<hostname>:<public-port>"`. The first draft of
  `parse_serve_status` treated *any* mapping proxying to a port other than
  the caller's own as offending, which made `tailnet` and `funnel` refuse
  each other the moment either was configured -- the opposite of "must run
  simultaneously". `parse_serve_status` now keys its "is this mapping
  stale" check on the one `Web` entry whose key ends `:{public_port}` (this
  adapter's own), and checks every entry, regardless of public port, only
  for the one thing that is unconditionally dangerous: a mapping that
  forwards straight into the port `local` holds -- the actual KI-17
  scenario the layer-2 check exists for.
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
  constraint on both fix rounds so far rules out. **The verification
  obligation this creates is stated in `base.py`'s `stop_command`
  docstring, not only here** -- `base.py` is the file `TransportManager`
  (Task 9) actually reads, since the whole point of the adapter pattern is
  that nothing outside `transports/` branches on which provider it is
  talking to (fix round 2, Critical).
- Both adapters' `stop_command` return an argv rather than `None`: `--bg`
  daemonises and the invoking process exits on its own, so killing it again
  touches nothing; only the `off` argv un-serves it.
- Both `command()` forms are the same shape (verb, `--bg`, `--https`,
  public port, local target URL) differing only in the verb and the public
  port.

`preflight()` **blocks the calling thread** (`subprocess.run`, a
`_PREFLIGHT_TIMEOUT_SECONDS` ceiling): if `TransportManager` (Task 9) calls
it from the event loop, it must wrap the call (e.g. `asyncio.to_thread`)
rather than discover the stall by running into it.

Layering: `io/api/` may import `core/` and `config` only -- but `config`
transitively reaches `llm` and `storage` (via `core.runtime_config`), so
importing it here would break `io.api never reaches past core and config`
despite the rule's letter (see `ui.py`'s closing comment for the same
landmine). This module avoids it: the port `local` holds is derived from
the *offset* arithmetic in `..listeners.LISTENER_OFFSETS` rather than from
`config.STUDIO_API_PORT` directly -- `preflight(port)` only ever receives
this transport's own already-resolved port, and `LISTENER_OFFSETS["local"]`
is always `0`, so `port - LISTENER_OFFSETS[self.name]` recovers the same
base port `local` binds to without a `config` import. `subprocess`, `json`,
`logging`, `re` and `urllib.parse` are stdlib; `..listeners` is a
zero-import sibling module. Nothing here reaches upward.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from urllib.parse import urlparse

from ..listeners import LISTENER_OFFSETS

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

# `tailscale {serve,funnel} status --json` is a local, already-running
# daemon query -- fast in practice. The timeout exists so a hung
# `tailscaled` cannot hang a transport start indefinitely; a timeout
# degrades to a warning exactly like any other unparseable output (see
# `_run_status`). `preflight()` blocks for up to this long.
_PREFLIGHT_TIMEOUT_SECONDS = 10.0

# A `*.ts.net` MagicDNS name: one or more dot-separated labels (letters,
# digits, internal hyphens) ending in the literal suffix `ts.net`. Anchored
# full-match against the *parsed* hostname (never the raw line) so
# "laptop.ts.net.evil.com" (suffix confusion), "evil.example.com" (wrong
# domain entirely) and "a.ts.net@evil.com" (userinfo confusion --
# `urlparse` resolves `.hostname` to `evil.com`, which then fails this
# match) all fail -- a name announced by the tunnel subprocess becomes a
# trusted `Host` and `Origin` (spec §8), so this must reject anything
# outside the provider's own shape rather than accept anything that merely
# looks like a hostname.
_TS_NET_HOSTNAME_RE = re.compile(
    r"^(?:[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+ts\.net$"
)

# The first `https://...` substring on a line of tunnel-process stdout, e.g.
# "Available within your tailnet: https://laptop.tail1234.ts.net/".
_URL_IN_LINE_RE = re.compile(r"https://\S+")


# ─── Preflight parsing (shared, KI-17 layer 2) ───────────────────────────────

def parse_serve_status(
    payload: dict,
    *,
    verb: str,
    public_port: int,
    target_port: int,
    local_port: int,
) -> str | None:
    """Reconcile a `tailscale {serve,funnel} status --json` document against
    the one mapping *this* adapter is about to claim.

    *verb* names which command produced *payload* (`"serve"` or
    `"funnel"`), used only to word the refusal in the right vocabulary --
    it never selects parsing behaviour, since both verbs describe the same
    underlying `Web` document (fix round 2, Important 1). *public_port* is
    the `--https` port this adapter's own mapping lives under (module
    constant per adapter); *target_port* is the local port this adapter
    forwards to (the *port* argument `preflight` received); *local_port* is
    the port the loopback `local` listener holds.

    Two independent things are checked, in order:

    1. **The actual KI-17 scenario, checked across every `Web` entry
       regardless of public port:** any mapping that proxies straight to
       *local_port* is refused unconditionally. This is what layer 2 exists
       to catch -- a `tailscale serve`/`funnel` configuration, however old
       or under whichever public port, that forwards public traffic into
       the loopback listener holding admin and `EXECUTE`.
    2. **This adapter's own mapping, and only its own:** the `Web` entry
       whose key ends `:{public_port}` -- if one exists and its proxy
       target is not *target_port*, that is refused too (a stale local
       target under a mapping this adapter itself is meant to own). A
       sibling transport's legitimate mapping, under a *different* public
       port, is never inspected by this step -- that is precisely the bug
       fix round 2 corrects: `tailnet` and `funnel` share one `Web`
       document but must not be able to refuse each other.

    Returns `None` when neither triggers (including the common case of no
    `Web` mappings at all -- confirmed against this machine's real,
    currently-clear `tailscale serve status --json`, which prints a bare
    `{}`). Refusal sentences name only port numbers and the corrective
    command -- never a hostname, a token or a path -- and never recommend
    `... reset`, which wipes every mapping on the device, not just this
    adapter's own.

    Degrades to a logged warning and `None` -- not a raised exception --
    on a shape, or a malformed `Proxy` port (out of range or non-numeric,
    `urlparse(...).port` raising `ValueError`), that this cannot recognise.
    Layer 3 (the per-listener `Host` gate) contains the failure either
    way; a preflight that hard-fails on an unrecognised Tailscale version
    would take the whole transport down for a formatting change.
    """
    try:
        web = payload.get("Web") or {}

        # 1. The actual KI-17 scenario -- unconditional, any public port.
        for key, mapping in web.items():
            handlers = mapping.get("Handlers") or {}
            for handler in handlers.values():
                proxy = handler.get("Proxy")
                if proxy is None:
                    logger.debug(
                        "tailscale %s status --json entry %r has a handler "
                        "with no recognised 'Proxy' target (a file/text "
                        "mapping, perhaps) -- skipped", verb, key,
                    )
                    continue
                if urlparse(proxy).port == local_port:
                    return (
                        f"tailscale {verb} already forwards a public "
                        f"mapping straight to port {local_port} -- that is "
                        f"the loopback listener's own port and must never "
                        f"be reachable through a tunnel; refusing to start "
                        f"until that mapping is corrected or removed"
                    )

        # 2. This adapter's own mapping, and only its own.
        our_suffix = f":{public_port}"
        for key, mapping in web.items():
            if not key.endswith(our_suffix):
                continue
            handlers = mapping.get("Handlers") or {}
            for handler in handlers.values():
                proxy = handler.get("Proxy")
                if proxy is None:
                    continue
                port = urlparse(proxy).port
                if port is not None and port != target_port:
                    return (
                        f"tailscale {verb} already has a mapping on public "
                        f"port {public_port} pointed at local port {port}, "
                        f"not this transport's own port {target_port} -- "
                        f"review 'tailscale {verb} status' and correct or "
                        f"remove just that mapping (e.g. 'tailscale {verb} "
                        f"--https {public_port} off'); 'tailscale {verb} "
                        f"reset' would also remove any other mappings "
                        f"configured on this device, not only this one"
                    )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "tailscale %s status --json produced a shape preflight does "
            "not recognise (%s); degrading to a warning rather than "
            "blocking the transport from starting", verb, exc,
        )
        return None

    return None


def _run_status(verb: str) -> dict | None:
    """Run `tailscale {verb} status --json` and parse it, or `None` on any
    failure to run or parse -- a missing binary, a timeout, or output that
    is not valid JSON all degrade the same way as an unrecognised shape."""
    try:
        result = subprocess.run(
            ["tailscale", verb, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning(
            "could not read 'tailscale %s status --json' (%s); degrading "
            "to a warning rather than blocking the transport from "
            "starting", verb, exc,
        )
        return None


# ─── Shared base ──────────────────────────────────────────────────────────────

class _TailscaleAdapterBase:
    """Hostname recognition and the KI-17 layer-2 preflight, shared by both
    Tailscale adapters. `command()`, `name` and `stop_command()` differ per
    subclass and are declared there; so do `_status_verb` (`"serve"` or
    `"funnel"`) and `_public_port`, which this base reads off the concrete
    subclass."""

    name: str
    _status_verb: str
    _public_port: int

    def hostname_from(self, line: str) -> str | None:
        match = _URL_IN_LINE_RE.search(line)
        if match is None:
            return None
        host = urlparse(match.group(0)).hostname
        if host is None or not _TS_NET_HOSTNAME_RE.fullmatch(host):
            return None
        return host

    def preflight(self, port: int) -> str | None:
        """Blocks the calling thread for up to `_PREFLIGHT_TIMEOUT_SECONDS`
        (`subprocess.run`) -- a caller on the event loop must wrap this
        (e.g. `asyncio.to_thread`) rather than discover the stall by
        running into it."""
        local_port = port - LISTENER_OFFSETS[self.name]
        payload = _run_status(self._status_verb)
        if payload is None:
            return None
        return parse_serve_status(
            payload,
            verb=self._status_verb,
            public_port=self._public_port,
            target_port=port,
            local_port=local_port,
        )


# ─── Tailnet adapter ──────────────────────────────────────────────────────────

class TailnetAdapter(_TailscaleAdapterBase):
    """`tailscale serve` -- reachable only by devices signed into the
    operator's own tailnet. The only transport in the system whose ceiling
    may ever be raised (spec §3), because reaching it already required a
    Tailscale login before a TENKA device credential was even presented."""

    name = "tailnet"
    _status_verb = "serve"
    _public_port = _TAILNET_PUBLIC_PORT

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
        removes.

        **The caller's verification obligation for this argv is stated in
        `base.py`'s `TransportAdapter.stop_command` docstring, not
        repeated here -- that is the file `TransportManager` reads.** In
        short: this was not exercised against a live mapping (both fix
        rounds so far were read-only), so the caller must re-read
        `tailscale serve status --json` after running this and confirm no
        mapping still targets port `8443` before treating the stop as
        successful."""
        return ["tailscale", "serve", "--https", str(_TAILNET_PUBLIC_PORT), "off"]


# ─── Funnel adapter ───────────────────────────────────────────────────────────

class FunnelAdapter(_TailscaleAdapterBase):
    """`tailscale funnel` -- the same machine and the same locally-terminated
    TLS as `tailnet`, but published to the open internet. One credential
    (the URL) instead of two. Never raisable (spec §3: `raisable=frozenset()`
    in `policy.py`)."""

    name = "funnel"
    _status_verb = "funnel"
    _public_port = _FUNNEL_PUBLIC_PORT

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
        local target URL). Built from the integer *port* and module
        constants only; a non-numeric *port* raises rather than reaching
        the argv (spec §8's subprocess-injection row)."""
        port = int(port)
        return [
            "tailscale", "funnel", "--bg", "--https", str(_FUNNEL_PUBLIC_PORT),
            f"http://127.0.0.1:{port}",
        ]

    def stop_command(self, port: int) -> list[str] | None:
        """`tailscale funnel --https 443 off` -- re-issues the same
        `--https` flag `command()` used, with `off` appended, mirroring
        `TailnetAdapter.stop_command`; *port* plays no part in which mapping
        `off` removes.

        **The caller's verification obligation for this argv is stated in
        `base.py`'s `TransportAdapter.stop_command` docstring, not repeated
        here.** In short: the caller must re-read `tailscale funnel status
        --json` after running this and confirm no mapping still targets
        port `443` before treating the stop as successful."""
        return ["tailscale", "funnel", "--https", str(_FUNNEL_PUBLIC_PORT), "off"]


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import transport_registry
    transport_registry.register("tailnet", TailnetAdapter())
    transport_registry.register("funnel", FunnelAdapter())


_register()
