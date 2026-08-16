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
  touches nothing -- undoing the mapping needs the separate `off` argv.
  Whether running that argv actually removed the mapping is exactly what
  is **not** assumed here (fix round 3, Must fix 2): see `base.py`'s
  `stop_command` docstring for the caller's verification obligation.
- Both `command()` forms are the same shape (verb, `--bg`, `--https`,
  public port, local target URL) differing only in the verb and the public
  port.
- **`AllowFunnel` on `tailnet`'s own port (fix round 3, fold-in).**
  `tailnet`'s public port, `8443`, is one of the three ports Tailscale
  Funnel itself is restricted to -- so a pre-existing or leftover
  `AllowFunnel` entry on that exact port would publish the `tailnet`
  listener to the open internet while TENKA believes it is tailnet-only.
  `tailnet` is the one transport whose ceiling is ever raisable
  (`EXECUTE`, `SYSTEM_CONTROL`), which makes this the highest-value single
  check `parse_serve_status` can make from a document it already parses.
  `TailnetAdapter` alone sets `_forbid_funnel = True`; `FunnelAdapter`
  leaves it `False` since funnel being funnelled is the point. **Fix round
  4** moved that check out of the `Web` loop: an `AllowFunnel` flag is a
  top-level key in its own right, so a *leftover* flag whose `Web` entry
  has already been removed -- the exact case the check was written for --
  was invisible while the check required a surviving mapping to hang off.
  TENKA's own subsequent `serve --https 8443` would then inherit it.
- **Preflight reads `tailscale serve status --json` for both adapters**
  (fix round 3 cheap fix). Both verbs return the same document on this
  machine's Tailscale 1.102.2, but that was never verified as a guarantee
  across versions -- `funnel status` might one day filter to
  funnel-enabled entries only, which would blind `FunnelAdapter`'s own
  KI-17 sweep to a serve-only mapping. Reading `serve status` for both
  costs nothing and removes the unverified assumption. `verb` (`"serve"`
  or `"funnel"`) is still adapter-specific and still selects only the
  *wording* of a refusal -- the command an operator would actually run to
  fix that adapter's own mapping. Fix round 4 finished the job: the
  status runner takes no verb at all any more (`_run_serve_status`), so
  the assumption cannot be re-introduced by a caller passing `"funnel"`.

`preflight()` **blocks the calling thread** (`subprocess.run`, a
`_PREFLIGHT_TIMEOUT_SECONDS` ceiling): if `TransportManager` (Task 9) calls
it from the event loop, it must wrap the call (e.g. `asyncio.to_thread`)
rather than discover the stall by running into it.

Layering: `io/api/` may import `core/` and `config` only -- but `config`
transitively reaches `llm` and `storage` (via `core.runtime_config`), so
importing it here would break `io.api never reaches past core and config`
despite the rule's letter (see `ui.py`'s closing comment for the same
landmine). This module avoids it: `preflight(port)` only ever receives
this transport's own already-resolved port, so `port - LISTENER_OFFSETS
[self.name]` recovers the base port without a `config` import -- but what
port `local` itself holds is then looked up through `..listeners.
local_port(base_port)` (imported here as `loopback_listener_port`, since
`parse_serve_status` takes a `local_port` parameter that would shadow the
bare name -- fix round 4 fold-in), the helper that exists for exactly
this, rather than assumed equal to the base port by a comment relying on
`LISTENER_OFFSETS["local"] == 0` (fix round 3 cheap fix: that assumption
now lives in code the reader can follow, in the one module that already
owns it). `subprocess`, `json`, `logging`, `re` and `urllib.parse` are
stdlib; `..listeners` is a zero-import sibling module. Nothing here
reaches upward.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from urllib.parse import urlparse

# Imported under an alias: `parse_serve_status` takes a keyword-only
# `local_port: int` parameter, and the bare name would be shadowed by it
# inside that function (fix round 4 fold-in). Harmless today only because
# `parse_serve_status` never calls the helper -- a trap for the next editor,
# so the two names are kept distinct rather than kept apart by luck.
from ..listeners import LISTENER_OFFSETS, local_port as loopback_listener_port

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
# `_run_serve_status`). `preflight()` blocks for up to this long.
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

def _public_port_from_key(key: object) -> str | None:
    """The public-port half of a `Web` dict key (`"<hostname>:<port>"`), or
    `None` if *key* is not that shape. Used only to name *which* mapping a
    refusal concerns -- never the hostname half (spec §2.3 L2: name the
    misconfiguration, never a hostname, a token or a path).

    The trailing half must be **all digits** (fix round 4 fold-in). Without
    that guard this returned whatever followed the last colon, so a key of
    an unexpected shape -- `"weird:host.ts.net"` -- put a *hostname* into a
    refusal sentence, which is precisely what this helper's own docstring
    and spec §2.3 L2 forbid. A non-numeric half is not a public port, so
    there is nothing here worth naming: the caller falls back to `"?"`.
    """
    if not isinstance(key, str) or ":" not in key:
        return None
    tail = key.rsplit(":", 1)[-1]
    if not tail.isdigit():
        return None
    return tail


def _proxy_port(proxy: object, *, verb: str, entry_port: str) -> int | None:
    """The local port a `Web` handler's `Proxy` target forwards to, or
    `None` when this handler carries no port this function can recognise.

    Both of `parse_serve_status`'s `Proxy`-reading loops go through here so
    they cannot drift apart in what they tolerate -- which is exactly how
    fix round 4's regression happened.

    **Tolerance is per-handler and never propagates** (fix round 4, Must
    fix). `Proxy` is documented as a URL string, but a status document is
    JSON: any scalar can appear there, and `urlparse` raises
    `AttributeError` -- not `ValueError` -- on every non-string it is
    handed (`int`, `float`, `bool`, `list`, `dict` all fail inside
    `urlparse`'s `_decode_args` with "object has no attribute 'decode'").
    Fix round 3 narrowed the caught exception to `ValueError` alone, which
    let a single non-string `Proxy` on *any* entry propagate out of
    `parse_serve_status` entirely, abandoning the rest of the KI-17 sweep
    -- including a sibling entry proxying straight into `local`'s port.

    Two defences, deliberately both: an explicit `isinstance(proxy, str)`
    guard states the contract in the code rather than leaving it implied by
    an exception list, and the `except` around the parse itself stays wide
    (`AttributeError`, `TypeError`, `ValueError`) so the tolerance does not
    depend on this module's enumeration of what `urlparse` can raise being
    complete. The guard is what a reader learns the contract from; the wide
    `except` is what keeps the sweep alive when the enumeration is wrong.
    """
    if proxy is None:
        logger.debug(
            "tailscale %s status --json entry on port %s has a handler "
            "with no recognised 'Proxy' target (a file/text mapping, "
            "perhaps) -- skipped",
            verb, entry_port,
        )
        return None
    if not isinstance(proxy, str):
        logger.warning(
            "tailscale %s status --json entry on port %s has a 'Proxy' "
            "value that is not a string (got %s) -- skipped, sweep "
            "continues",
            verb, entry_port, type(proxy).__name__,
        )
        return None
    try:
        return urlparse(proxy).port
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "tailscale %s status --json entry on port %s has an "
            "unparseable Proxy port (%s) -- skipped, sweep continues",
            verb, entry_port, exc,
        )
        return None


def parse_serve_status(
    payload: dict,
    *,
    verb: str,
    public_port: int,
    target_port: int,
    local_port: int,
    forbid_funnel: bool = False,
) -> str | None:
    """Reconcile a `tailscale serve status --json` document against the one
    mapping *this* adapter is about to claim.

    *verb* names which command an operator would run to fix *this*
    adapter's own mapping (`"serve"` or `"funnel"`), used only to word a
    refusal in the right vocabulary -- it never selects parsing behaviour,
    since `serve status --json` describes both verbs' mappings in one `Web`
    document (fix round 2, Important 1). *public_port* is the `--https`
    port this adapter's own mapping lives under (module constant per
    adapter); *target_port* is the local port this adapter forwards to (the
    *port* argument `preflight` received); *local_port* is the port the
    loopback `local` listener holds. *forbid_funnel* is `True` only for
    `tailnet` (fix round 3 fold-in): `tailnet`'s public port, `8443`, is one
    of Funnel's three permitted ports, so a stray `AllowFunnel` entry there
    would publish the one raisable listener to the open internet.

    Three independent things are checked, each tolerant of a malformed
    individual `Web` entry (fix round 3, Must fix 1 -- an entry this cannot
    parse is skipped with a warning and the sweep continues to the next
    entry; it never aborts the whole check and returns a false "clear",
    which is what a single `try` around the entire sweep did before this
    fix, and which the `ValueError` added for a malformed `Proxy` port
    would otherwise have made easier to trigger, not harder). Every
    `Proxy` value goes through `_proxy_port`, which owns that tolerance for
    both loops at once so they cannot drift apart in what they survive --
    fix round 3 narrowed one of them to `ValueError` alone and a non-string
    `Proxy` (which raises `AttributeError` from `urlparse`) then propagated
    out of this function entirely, abandoning the sweep before it reached
    the dangerous entry (fix round 4, Must fix):

    1. **The actual KI-17 scenario, checked across every `Web` entry
       regardless of public port:** any mapping that proxies straight to
       *local_port* is refused unconditionally, naming which public port
       the offending mapping was found under (never its hostname). This is
       what layer 2 exists to catch -- a `tailscale serve`/`funnel`
       configuration, however old or under whichever public port, that
       forwards public traffic into the loopback listener holding admin
       and `EXECUTE`.
    2. **This adapter's own public port must not be marked `AllowFunnel`
       when *forbid_funnel* is set:** the `AllowFunnel` document is read
       directly for any key ending `:{public_port}`, **independently of
       whether a `Web` entry still exists under that key** (fix round 4
       fold-in -- nesting this inside the `Web` loop, as fix round 3 did,
       missed the "leftover flag, mapping already removed" half of the very
       threat it was written for, and TENKA's own subsequent `serve --https
       {public_port}` would inherit that flag).
    3. **This adapter's own mapping's proxy target:** if the `Web` entry
       keyed under `:{public_port}` has a proxy target that is not
       *target_port*, that is refused too (a stale local target under a
       mapping this adapter itself is meant to own).
       A sibling transport's legitimate mapping, under a *different*
       public port, is never inspected by checks 2 or 3 -- that is
       precisely the bug fix round 2 corrects: `tailnet` and `funnel`
       share one `Web` document but must not be able to refuse each other.

    Returns `None` when nothing triggers (including the common case of no
    `Web` mappings at all -- confirmed against this machine's real,
    currently-clear `tailscale serve status --json`, which prints a bare
    `{}`). Refusal sentences name only port numbers and the corrective
    command -- never a hostname, a token or a path -- and never recommend
    `... reset` as *the* fix, since that wipes every mapping on the device,
    not just this adapter's own (though a refusal may still name `reset`
    while warning what it would destroy).

    Degrades a single unrecognised `Web` entry to a logged warning and
    moves on to the next one; degrades to `None` outright only when
    *payload* or its `Web` value is not the documented shape at all (there
    is then nothing to sweep). Layer 3 (the per-listener `Host` gate)
    contains the failure either way; a preflight that hard-fails on an
    unrecognised Tailscale version would take the whole transport down for
    a formatting change.
    """
    if not isinstance(payload, dict):
        logger.warning(
            "tailscale %s status --json produced a shape preflight does "
            "not recognise (expected a JSON object, got %s); degrading to "
            "a warning rather than blocking the transport from starting",
            verb, type(payload).__name__,
        )
        return None

    web = payload.get("Web")
    if web is None:
        web = {}
    if not isinstance(web, dict):
        logger.warning(
            "tailscale %s status --json's 'Web' key is not the documented "
            "shape (expected an object, got %s); degrading to a warning "
            "rather than blocking the transport from starting",
            verb, type(web).__name__,
        )
        return None

    allow_funnel = payload.get("AllowFunnel")
    if not isinstance(allow_funnel, dict):
        allow_funnel = {}

    # 1. The actual KI-17 scenario -- unconditional, any public port, and
    # tolerant per-entry so one unrecognised mapping cannot blind the sweep
    # to the rest of the document.
    for key, mapping in web.items():
        entry_port = _public_port_from_key(key) or "?"
        if not isinstance(mapping, dict):
            logger.warning(
                "tailscale %s status --json entry on port %s is not the "
                "documented shape -- skipped, sweep continues",
                verb, entry_port,
            )
            continue
        handlers = mapping.get("Handlers")
        if not isinstance(handlers, dict):
            continue
        for handler in handlers.values():
            if not isinstance(handler, dict):
                continue
            proxy_port = _proxy_port(
                handler.get("Proxy"), verb=verb, entry_port=entry_port,
            )
            if proxy_port == local_port:
                return (
                    f"tailscale {verb} already forwards a public mapping "
                    f"on port {entry_port} straight to port {local_port} "
                    f"-- that is the loopback listener's own port and "
                    f"must never be reachable through a tunnel; refusing "
                    f"to start until that mapping is corrected or removed"
                )

    our_suffix = f":{public_port}"

    # 2. AllowFunnel on our own public port -- read from the `AllowFunnel`
    # document *directly*, never from inside the `Web` loop (fix round 4
    # fold-in). Fix round 3 nested this check under a surviving `Web` entry,
    # which covered only half its own threat: the "leftover" case it was
    # written for is an `AllowFunnel` flag whose `Web` entry is already gone,
    # and TENKA's own subsequent `serve --https {public_port}` would then
    # inherit the flag. `AllowFunnel` is its own top-level key, so checking
    # it independently costs one loop and covers both halves.
    if forbid_funnel:
        for key, funnelled in allow_funnel.items():
            if not isinstance(key, str) or not key.endswith(our_suffix):
                continue
            if not funnelled:
                continue
            return (
                f"tailscale {verb}'s own public port {public_port} is "
                f"marked AllowFunnel -- refusing to start until Funnel is "
                f"disabled for that port ('tailscale funnel --https "
                f"{public_port} off'); a leftover flag counts, with or "
                f"without a mapping still under it"
            )

    # 3. This adapter's own mapping's proxy target, and only its own.
    for key, mapping in web.items():
        if not isinstance(key, str) or not key.endswith(our_suffix):
            continue
        if not isinstance(mapping, dict):
            continue

        handlers = mapping.get("Handlers")
        if not isinstance(handlers, dict):
            continue
        for handler in handlers.values():
            if not isinstance(handler, dict):
                continue
            port = _proxy_port(
                handler.get("Proxy"), verb=verb, entry_port=str(public_port),
            )
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

    return None


def _run_serve_status() -> dict | None:
    """Run `tailscale serve status --json` and parse it, or `None` on any
    failure to run or parse -- a missing binary, a timeout, or output that
    is not valid JSON all degrade the same way as an unrecognised shape.

    Takes no verb (fix round 4 fold-in). Fix round 3 left this
    verb-parameterised for generality after making both adapters call it
    with `"serve"`, and generality nothing exercises is a trap here rather
    than a convenience: the one other value it would accept, `"funnel"`, is
    exactly the call that could one day return a funnel-filtered document
    and blind `FunnelAdapter`'s own KI-17 sweep to a serve-only mapping --
    the assumption the round-3 fix removed. Dropping the parameter makes
    "both adapters read the one unfiltered document" structural instead of
    conventional. An adapter that genuinely needs a different verb should
    add its own function and say why, not re-widen this one.
    """
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
            "from starting", exc,
        )
        return None


# ─── Shared base ──────────────────────────────────────────────────────────────

class _TailscaleAdapterBase:
    """Hostname recognition and the KI-17 layer-2 preflight, shared by both
    Tailscale adapters. `command()`, `name` and `stop_command()` differ per
    subclass and are declared there; so do `_status_verb` (`"serve"` or
    `"funnel"` -- wording only, see `preflight` below), `_public_port` and
    `_forbid_funnel`, which this base reads off the concrete subclass."""

    name: str
    _status_verb: str
    _public_port: int
    _forbid_funnel: bool = False

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
        running into it.

        Always reads `tailscale serve status --json`, for both adapters
        (fix round 3 cheap fix) -- both verbs return the same document on
        the installed 1.102.2 binary, and reading the one status call for
        both removes an unverified assumption about `funnel status`'s
        shape rather than relying on it. `_status_verb` still selects only
        the *wording* `parse_serve_status` uses in a refusal."""
        base_port = port - LISTENER_OFFSETS[self.name]
        local = loopback_listener_port(base_port)
        payload = _run_serve_status()
        if payload is None:
            return None
        return parse_serve_status(
            payload,
            verb=self._status_verb,
            public_port=self._public_port,
            target_port=port,
            local_port=local,
            forbid_funnel=self._forbid_funnel,
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
    # The one raisable ceiling in the system (EXECUTE, SYSTEM_CONTROL) --
    # `8443` is one of Funnel's three permitted ports, so a stray
    # `AllowFunnel` entry there must refuse the start, not pass silently.
    _forbid_funnel = True

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
        short: this was not exercised against a live mapping (all three
        fix rounds so far were read-only), so the caller must re-read
        `tailscale serve status --json` after running this and confirm no
        `Web` entry is still keyed under public port `8443` before
        treating the stop as successful."""
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
    # Never raisable (policy.py: raisable=frozenset()) and funnel being
    # funnelled is the point -- unlike `tailnet`, an AllowFunnel entry on
    # this adapter's own port is expected, not a KI-17-shaped surprise.
    _forbid_funnel = False

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
        here.** In short: the caller must re-read `tailscale serve status
        --json` (the same document `preflight` reads for both adapters,
        for the same reason -- module docstring's "cheap fix" note) after
        running this and confirm no `Web` entry is still keyed under
        public port `443` before treating the stop as successful."""
        return ["tailscale", "funnel", "--https", str(_FUNNEL_PUBLIC_PORT), "off"]


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import transport_registry
    transport_registry.register("tailnet", TailnetAdapter())
    transport_registry.register("funnel", FunnelAdapter())


_register()
