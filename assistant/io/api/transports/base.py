# assistant/io/api/transports/base.py
"""The transport adapter contract.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §4: a
transport is a mechanism (Tailscale, Cloudflare, ... whatever comes next), and
what each one *is* stays declarative. An adapter declares, and nothing else:

- its policy name (and therefore, via spec §2.2, its port);
- the argv to spawn for a given port;
- how to recognise its public hostname in the process output;
- its preflight check, if any (spec §2.3 L2);
- how to stop (`stop_command`).

Nothing outside `transports/` branches on which transport it is talking to --
that is what makes a fourth provider a new module and nothing else, the same
shape `llm/providers/` and `tool_registry` already use elsewhere in this tree.

Layering: `io/api/` may import `core/` and `config` only. Importing `..policy`
from inside `transports/` is legal and expected -- a transport's own policy
name has to be checked against `POLICIES` somewhere, and the registry
(`__init__.py`) is that somewhere, not this module.
"""
from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# A tunnel subprocess that starts but never announces a hostname does not
# serve (spec §4): the listener is torn down and the subprocess killed rather
# than left half-open forever. This is the hard ceiling on that wait, shared
# by every adapter so none of them can quietly wait longer than the others.
HOSTNAME_TIMEOUT_SECONDS = 30.0


# ─── Transport Adapter Protocol ──────────────────────────────────────────────

@runtime_checkable
class TransportAdapter(Protocol):
    """What a transport provider (Tailscale, Cloudflare, ...) must declare.

    `name` doubles as the adapter's `POLICIES` key and, via
    `port_for(name, base_port)` (Task 3), its fixed port -- the registry
    (`TransportRegistry.register`) is what checks that the name is actually
    one of `POLICIES` and is never `"local"`; this Protocol only pins the
    shape every adapter must have to be registerable at all.
    """

    name: str

    def command(self, port: int) -> list[str]:
        """The argv to spawn for *port*, built from the integer and module
        constants only. Spec §2.3 L1 / §8: no caller-supplied string may
        reach this -- the operator never types a port, and TENKA builds the
        whole command line itself."""
        ...

    def hostname_from(self, line: str) -> str | None:
        """Recognise this provider's own public hostname in one line of the
        spawned process's output, or return `None` if *line* carries none.
        A name this returns becomes a trusted `Host` and `Origin`
        (`PublishedHosts.publish`), so an adapter must reject a shape outside
        its own domain rather than accept anything that merely looks like a
        hostname."""
        ...

    def public_port(self) -> int:
        """The public (`--https`) port this adapter's tunnel is reachable on
        -- `8443` for `tailnet`, `443` for `funnel` and `quick` (module
        constants in `tailscale.py` / `cloudflare.py`, the same ones
        `command()` builds its `--https` flag from).

        The one integer every URL/origin site built from a published
        hostname needs and none of them owns: `public_url` below folds it
        into a string for callers inside `transports/`, and
        `TransportManager._publish` (Task 9) passes this same value into
        `PublishedHosts.publish(..., public_port=...)` so that
        `security.py` -- which must not import `transports/` -- can build
        the identical port-carrying origin from data alone, via
        `PublishedHosts.origins_for`. Both readers derive from this one
        method so the two cannot drift apart the way a separately-declared
        constant in each place could."""
        ...

    def public_url(self, hostname: str) -> str:
        """The reachable `https://` URL for *hostname*, this adapter's own
        public port (`public_port()`) folded in.

        `hostname` (from `hostname_from`, trusted and bare -- see
        `PublishedHosts`) is never the whole story: Tailscale keys a
        serve/funnel mapping on a public port (`--https`, module constant
        per adapter in `tailscale.py`) that is not always the HTTPS default,
        and nothing about the hostname itself says which port a client must
        connect to. `tailnet` publishes on `8443`; a URL built as bare
        `https://{hostname}` is one a browser or a phone resolves to 443,
        where nothing is listening -- `ERR_CONNECTION_TIMED_OUT`, not a
        reachable Studio.

        This is the one place port knowledge belongs: each adapter already
        owns its own public port (`command()` builds the `--https` flag from
        the same constant), so each adapter builds its own URL from it too,
        rather than a caller outside `transports/` branching on which
        provider it is talking to. Only the *hostname* stored in
        `PublishedHosts` stays bare -- `host_is_allowed` matches on it with
        the port deliberately stripped (`hostname_of`), because a tunnel
        forwards the public authority and its port is not the daemon's; this
        method reads that same bare hostname and adds a port only in the
        string it returns, never in what gets published or matched against.

        A provider whose public port is always the HTTPS default (`quick`,
        plain 443; `funnel`, 443 by the port-split in `tailscale.py`) omits
        `:443` -- a URL with an explicit default port is ugly and some
        clients normalise it away anyway."""
        ...

    def preflight(self, port: int) -> str | None:
        """Reconcile with the provider's own persisted state before binding,
        if the provider has any to reconcile with (spec §2.3 L2). Returns
        `None` when clear to start, or a refusal sentence naming the
        offending pre-existing mapping when not. A provider with nothing to
        reconcile (a quick tunnel has no persisted configuration) always
        returns `None` and says so in its docstring, rather than leaving the
        reader to wonder whether the check was forgotten."""
        ...

    def stop_command(self, port: int) -> list[str] | None:
        """The argv of a second command needed to undo what `command()`
        started, or `None` if none is needed. This method only *names* a
        command -- it does not run anything or perform the stop itself;
        running it (or, when `None`, simply terminating the tracked
        subprocess) is the caller's job, exactly like `command()` names
        the start argv without spawning it.

        `None` means the spawned subprocess's own lifetime *is* the
        tunnel's lifetime, so terminating it is enough (a long-running
        foreground process, e.g. `cloudflared tunnel --url ...`). A
        returned argv is required to undo a provider whose spawn already
        detached -- `tailscale serve --bg` daemonises and its invoking
        process exits on its own, so killing that process again touches
        nothing.

        **Running this argv is not, by itself, proof the tunnel is down.**
        The Tailscale adapters' argv (an `... off` form) is Tailscale's own
        documented way to remove one mapping, but it was verified against
        Tailscale's documentation, not by execution against a live mapping
        -- and the same `<target>` grammar `command()` uses accepts a bare
        file, directory or arbitrary text, so a subtly wrong invocation
        could be silently reinterpreted as a new thing to serve rather than
        a request to stop serving: the process would exit 0 while the
        mapping stayed up. Any adapter whose `stop_command` returns a
        non-`None` argv for the same reason (its provider daemonises and
        only an explicit second command undoes it) carries the same
        obligation. The caller **MUST** treat a returned argv's exit code
        as provisional: run it, then re-read the provider's own status
        (e.g. `tailscale {serve,funnel} status --json`) and confirm no
        `Web` entry is still keyed under this transport's public port,
        before treating the stop as successful -- not "no mapping still
        targets" that port, a phrasing that would always be true here
        since a mapping *targets* a local port and is *keyed under* a
        public one, and so would let a literal reading conclude every stop
        succeeded (fix round 3, Must fix 2: this is exactly the wording
        that leaked the Critical this docstring exists to prevent, the
        first time it was written). If a `Web` entry is still keyed under
        that public port, fail loudly rather than report success -- an
        internet-facing listener that silently stayed up is the worst
        failure mode this milestone can produce, worse than a stop that
        visibly failed.

        Note what that obligation costs the caller: **nothing on this
        Protocol yields the public port it names.** `name`, `command`,
        `hostname_from`, `preflight` and `stop_command` are the whole
        surface, and `port` throughout means the *local* port a transport
        forwards to, never the public one it is published on -- so the
        manager must recover the public port from the argv this method
        returns (the value following `--https`), which is the only place
        an adapter states it. That is deliberate: the port to check is
        then read off the same command whose effect is being checked, so
        the two cannot drift apart the way a separately-declared constant
        could. See `transports/tailscale.py`'s two adapters for the
        concrete case this obligation exists for."""
        ...

    def status_command(self, port: int) -> list[str] | None:
        """The argv that re-reads this provider's own state, so a caller can
        verify a stop actually stopped -- or `None` when there is nothing to
        read back.

        Split out from `stop_command` (fix round 1, Minor 3) because the
        caller must not guess it. The first version of `TransportManager`
        derived the reader from the stop argv, as `<binary> <verb> status
        --json`, which assumed every provider's CLI has that shape and --
        worse -- silently chose `funnel status` over `serve status` for the
        Funnel adapter. Those two return the same document on the Tailscale
        binary this was verified against, but that is not a guarantee across
        versions, and an adapter that has already decided which document its
        `preflight` trusts is the only place that decision belongs.

        An adapter whose `stop_command` returns an argv **must** return one
        here too: the pair is what makes a stop verifiable, and a caller that
        cannot verify must report the stop as unverified rather than as
        successful. An adapter whose `stop_command` is `None` has nothing to
        verify (reaping the spawned process *is* its stop) and returns `None`
        here as well, saying so in its docstring rather than leaving the
        reader to wonder.
        """
        ...


# ─── Transport Session ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransportSession:
    """One running instance of a transport -- the bookkeeping
    `TransportManager` (Task 9) needs to unwind a start in exact reverse
    (spec §4's start/stop sequence): unpublish the owner, drop the registry
    entry, drop every raise scoped to the policy, cancel the serve task,
    close the socket, terminate and reap the subprocess.

    `owner` is a fresh id per session, never the transport's name -- two
    successive runs of the same tunnel get different names from the
    provider, so trusting the second must not leave the first's name
    published (the hostname-reuse class `PublishedHosts` exists to prevent).

    Frozen because a session's identity (which port, whose subprocess, which
    socket) never changes once started; `hostname` is filled in only after
    the announced name is read under `HOSTNAME_TIMEOUT_SECONDS`, so it starts
    `None` and a session with a discovered hostname is a new value
    (`dataclasses.replace`), not a mutation of this one.
    """

    policy_name: str
    port: int
    owner: str
    process: asyncio.subprocess.Process
    sock: socket.socket
    serve_task: asyncio.Task
    hostname: str | None = None
    # The adapter this session was started with, carried here so a stop runs
    # *its* stop and status commands rather than whatever the registry holds
    # by that name later (fix round 1, Minor 8). Optional only so that
    # existing constructions of this dataclass stay valid; a session the
    # manager built always has one, and a stop that finds `None` reports an
    # unverified stop instead of guessing.
    adapter: "TransportAdapter | None" = None

    @property
    def url(self) -> str | None:
        """The session's public URL, or `None` before a hostname has been
        announced. Spec §5.3-§5.4 fix the scheme as `https://` for all three
        tunnels -- Tasks 12 (QR encoding) and 13 (an API payload) both need
        exactly this string, so it is derived once here rather than in two
        places that could drift apart.

        The public *port* is the adapter's own knowledge (`public_url` on
        `TransportAdapter`), not this dataclass's -- `tailnet` publishes on
        `8443`, not the HTTPS default, and a bare `https://{hostname}` is a
        URL that resolves to a port nothing is listening on. Delegated to
        `self.adapter` when one is carried (every session `TransportManager`
        actually starts carries one); a session built without one -- test
        bookkeeping only, never a real start -- falls back to the bare
        default-port form, since that was this property's only shape before
        the public port existed anywhere."""
        if self.hostname is None:
            return None
        if self.adapter is not None:
            return self.adapter.public_url(self.hostname)
        return f"https://{self.hostname}"
