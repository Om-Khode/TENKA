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
        visibly failed. See `transports/tailscale.py`'s two adapters for
        the concrete case this obligation exists for."""
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

    @property
    def url(self) -> str | None:
        """The session's public URL, or `None` before a hostname has been
        announced. Spec §5.3-§5.4 fix the scheme as `https://` for all three
        tunnels -- Tasks 12 (QR encoding) and 13 (an API payload) both need
        exactly this string, so it is derived once here rather than in two
        places that could drift apart."""
        if self.hostname is None:
            return None
        return f"https://{self.hostname}"
