# assistant/io/api/transports/cloudflare.py
"""The Cloudflare transport adapter -- `quick`, a `cloudflared` unnamed tunnel.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.3, §4, §8.

**Why this transport is trusted with almost nothing.** `quick` is different in
kind from the two Tailscale transports, and the difference is not reach but
who reads the bytes. Tailscale terminates TLS on this machine, so its relays
carry ciphertext; Cloudflare terminates TLS at its own edge and forwards
plaintext down the tunnel, which makes **Cloudflare a permitted MITM** on
everything this listener carries (spec §8's MITM row). Three consequences,
all of them already written into `..policy.POLICIES["quick"]` and none of
them this module's to decide:

- its ceiling is `OBSERVE` alone -- watching her work, never acting and never
  reading what she has stored;
- its `raisable` set is empty, so the ceiling can never be lifted: a raise
  widens what a raised device may do, not who else can read it, and here a
  third party reads it regardless;
- pairing is refused on it (Task 12) -- minting a device credential in front
  of a party who can read the exchange is not a thing to do.

There is a fourth consequence that is not a policy value: a `*.trycloudflare.
com` name is recycled to other users once the tunnel ends, so a name left
trusted after a stop is a year-long host-only cookie pointed at a stranger's
access log (spec §8's hostname-reuse row). `PublishedHosts` ownership is the
mechanism; this module only has to make sure the name it announces is
genuinely one of Cloudflare's.

**Verified command and output shape.** `cloudflared` is **not installed on
this machine** -- unlike Task 7's Tailscale adapters, nothing here was
confirmed against a running binary, and the live-test task is what installs
it and checks. What is written here comes from Cloudflare's own current
documentation:

- The command form, verbatim from
  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/
  and https://developers.cloudflare.com/pages/how-to/preview-with-cloudflare-tunnel/
  -- `cloudflared tunnel --url http://localhost:8080`. No account, no
  credentials file, no named tunnel: "`cloudflared` will generate a random
  subdomain when connecting to the Cloudflare network and print it in the
  terminal for you to use and share."
- The announcement, printed inside an ASCII box. Cloudflare's Pages guide
  shows the older wording:

      2021-07-15T20:11:32Z INF +------------------------------------------+
      2021-07-15T20:11:32Z INF |  Your free tunnel has started! Visit it:  |
      2021-07-15T20:11:32Z INF |    https://<name>.trycloudflare.com       |
      2021-07-15T20:11:32Z INF +------------------------------------------+

  and current `cloudflared` releases word the same box "Your quick Tunnel has
  been created! Visit it at (it may take some time to be reachable):".
  `hostname_from` deliberately keys on **the URL alone and never on the
  surrounding prose**, precisely because that prose has already changed once
  between releases while the URL has not.
- **The banner goes to stderr, not stdout.** `cloudflared` logs everything
  through its logger, which writes to stderr by default. The manager (Task 9)
  reading only stdout would wait out `HOSTNAME_TIMEOUT_SECONDS` and tear a
  perfectly healthy tunnel down -- it must read stderr, or merge the two.

**`--http-host-header` is never emitted, and that is a security property, not
a preference.** The flag rewrites the `Host` header `cloudflared` forwards.
KI-17's layer 3 -- the load-bearing layer, spec §2.3 -- is that each listener
gates on `Host`, with `local` accepting loopback names only, so a tunnelled
request arriving on the local port carries the public authority and is
answered 421 before authentication, before policy lookup, before any route
runs. `cloudflared --http-host-header 127.0.0.1:8787` rewrites `Host` to a
loopback name and walks straight through it. Spec §2.3 records that as a
stated gap requiring an attacker who is already executing processes on this
machine; what is fully in TENKA's own hands is that *TENKA* never emits it,
and `test_the_quick_command_never_sets_http_host_header` is what keeps a
future fix for a `Host`-related bug from reaching for the flag that defeats
the milestone.

Layering: `io/api/` may import `core/` and `config` only. This module needs
neither -- `re` and `urllib.parse` are stdlib and nothing else is imported at
all, not even the `..listeners` helpers Task 7's adapters need for their
preflight, since `quick` has no preflight. Nothing here reaches upward.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# A `*.trycloudflare.com` quick-tunnel name: one or more dot-separated labels
# (letters, digits, internal hyphens) ending in the literal suffix
# `trycloudflare.com`. Anchored full-match against the *parsed* hostname,
# never against the raw line -- the same shape Task 7's `_TS_NET_HOSTNAME_RE`
# uses, and for the same reason: a name announced by the tunnel subprocess
# becomes a trusted `Host` and `Origin` (spec §8), so this must reject
# anything outside Cloudflare's own domain rather than accept whatever merely
# looks like a hostname. Parsing first and full-matching the extracted host
# is what makes the three standard confusions fail --
# "a.trycloudflare.com.evil.com" (suffix), "a.nottrycloudflare.com" (prefix:
# the label before the suffix must be terminated by a literal dot), and
# "https://a.trycloudflare.com@evil.com/" (userinfo: `urlparse` resolves
# `.hostname` to "evil.com", which then fails this match).
#
# At least one leading label is required, so the bare apex `trycloudflare.com`
# is rejected: a quick tunnel is always a generated subdomain, and the apex is
# Cloudflare's own site rather than this machine.
_TRYCLOUDFLARE_HOSTNAME_RE = re.compile(
    r"^(?:[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+trycloudflare\.com$"
)

# The first `https://...` substring on a line of tunnel-process output. Only
# the *first* is considered, so a hostile or confused line that appends a
# legitimate name after a bad one ("https://evil.com/ https://a.trycloudflare
# .com/") is rejected on the bad one rather than rescued by the good one.
#
# `\S+` stops at whitespace, which is what the ASCII box guarantees: the box
# is sized to its longest line (the prose), so the shorter URL line is always
# padded with spaces before the closing `|`. Widening this to also stop at
# `|` was considered and rejected -- it would accept
# "https://a.trycloudflare.com|@evil.com", which `\S+` correctly resolves to
# the host "evil.com" and refuses. An unpadded box would make the hostname
# undiscoverable and the transport would decline to serve
# (`HOSTNAME_TIMEOUT_SECONDS`), which is the right way to be wrong here.
_URL_IN_LINE_RE = re.compile(r"https://\S+")


# ─── Quick adapter ───────────────────────────────────────────────────────────

class QuickAdapter:
    """`cloudflared tunnel --url` -- a throwaway public URL, reachable by
    anyone who holds the link, with Cloudflare reading the plaintext in
    between. `OBSERVE` alone, never raisable, never paired on (module
    docstring)."""

    name = "quick"

    def command(self, port: int) -> list[str]:
        """`cloudflared tunnel --url http://127.0.0.1:{port}` -- and nothing
        else. Built from the integer *port* and module-level literals only;
        a non-numeric *port* raises rather than reaching the argv (spec §8's
        subprocess-injection row).

        No credentials file, no tunnel name, no config: an unnamed tunnel
        takes none, which is the whole reason `quick` needs no preflight.

        **Never `--http-host-header`.** That flag rewrites the `Host` header
        `cloudflared` forwards, and KI-17's load-bearing layer 3 is that a
        listener gates on `Host` -- see the module docstring. The argv is
        pinned by `test_the_quick_command_never_sets_http_host_header`.
        """
        port = int(port)
        return ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"]

    def hostname_from(self, line: str) -> str | None:
        """The `*.trycloudflare.com` name announced on *line*, or `None`.

        Keys on the URL alone and never on the surrounding banner prose,
        which has already changed once between `cloudflared` releases while
        the URL has not (module docstring). The returned name becomes a
        trusted `Host` and `Origin`, so the extracted host is full-matched
        against Cloudflare's own domain shape -- anything else, including a
        legitimate name that merely appears after a bad one on the same
        line, yields `None`.
        """
        match = _URL_IN_LINE_RE.search(line)
        if match is None:
            return None
        host = urlparse(match.group(0)).hostname
        if host is None or not _TRYCLOUDFLARE_HOSTNAME_RE.fullmatch(host):
            return None
        return host

    def preflight(self, port: int) -> str | None:
        """Always `None`: there is **no persisted configuration** for an
        unnamed tunnel to have gone stale, so there is nothing to reconcile.

        This is stated rather than left blank so a reader can tell the check
        was considered and found inapplicable, not forgotten (`base.py`'s
        `TransportAdapter.preflight` docstring asks for exactly that). Spec
        §2.3 L2 exists for the real-world Tailscale case: a `tailscale serve`
        mapping the operator set up by hand months ago, still living in
        Tailscale's own state, still pointed at 8787. `cloudflared tunnel
        --url` has no equivalent -- it reads no credentials file, claims no
        named tunnel, writes nothing to disk, and Cloudflare mints the
        hostname fresh per run and forgets it when the process exits. There
        is no prior mapping for a stale one to be, and therefore no state
        for this adapter to read back and refuse on.

        `quick` is not left undefended by that. L1 still holds -- TENKA
        builds this argv itself from registry data, and the operator never
        types a port -- and L3, the load-bearing layer, holds regardless of
        which tunnel software is involved or whether TENKA launched it at
        all.

        Unlike the Tailscale adapters' `preflight`, this **runs no
        subprocess and does not block**; a caller on the event loop needs no
        `asyncio.to_thread` wrapper for it (though wrapping every adapter's
        uniformly is fine, since a caller must not branch on which transport
        it is talking to).
        """
        return None

    def status_command(self, port: int) -> list[str] | None:
        """`None` -- there is nothing to read back.

        An unnamed quick tunnel has no persisted configuration (the same
        reason `preflight` is `None`), and `stop_command` is `None` too, so
        there is no second command whose effect would need verifying: the
        tunnel is up exactly as long as the spawned process is, and reaping
        that process is both the stop and the proof of it. Stated rather than
        left blank so a reader can tell it was considered
        (`base.py`'s `status_command` docstring asks for exactly that).
        """
        return None

    def stop_command(self, port: int) -> list[str] | None:
        """`None` -- terminating the spawned process *is* the stop.

        `cloudflared tunnel --url` runs in the **foreground**: it holds the
        connection to Cloudflare's edge for as long as it lives, and the
        hostname it was given dies with it. That is the `None` case
        `base.py`'s `stop_command` docstring describes, and it is the
        opposite of the Tailscale adapters -- they spawn with `--bg`, the
        invoking process daemonises and exits on its own, so killing it
        again touches nothing and only an explicit `off` argv undoes the
        mapping.

        The caller therefore has no second command to run and no exit code
        to distrust here; what it must still do is reap the process and
        confirm it is actually gone, since the tunnel is up exactly as long
        as that process is.
        """
        return None


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import transport_registry
    transport_registry.register("quick", QuickAdapter())


_register()
