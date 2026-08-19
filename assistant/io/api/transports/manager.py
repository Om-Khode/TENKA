# assistant/io/api/transports/manager.py
"""The transport lifecycle -- start one, stop one, and leave nothing behind.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.3 (L1),
§4. This is the seam: `base.py` says what an adapter declares, `tailscale.py`
and `cloudflare.py` declare it, `listeners.py` says which port each policy
holds, `server.py` binds and serves a socket, `security.PublishedHosts` says
which hostnames are trusted and `raises.RaiseStore` holds the ceiling raises.
Nothing here branches on *which* transport it is talking to -- that is the
whole point of the Protocol, and it is what makes a fourth provider a new
module and nothing else.

**Start, in this order, each step undoing the previous ones on failure:**
preflight -> bind the socket -> register the port in the *live*
`listener_policies` registry -> spawn the subprocess with the adapter's own
argv -> read the announced hostname under `HOSTNAME_TIMEOUT_SECONDS` ->
`publish(host, owner=session_id, listener=port)` -> serve.

Preflight runs *before* the bind (rather than after, as spec §4's prose has
it) because a refusal is the expected outcome of a misconfigured machine, and
a refusal that has already bound a port and written a registry entry has to
unwind two things to say "no". Nothing downstream of preflight depends on the
socket existing.

**Stop and crash run one shared teardown**, idempotent, in reverse: unpublish
the owner, drop the registry entry, `raises.drop_policy(name)`, cancel the
serve task, close the socket, run `stop_command`'s argv *and verify the
mapping is gone*, terminate and reap the subprocess. `TransportManager.stop`
is literally what a crash handler calls, so "the same teardown" is one method
rather than two that must be kept in step.

Three things in here exist because they were got wrong somewhere first:

1. **Both streams are scanned for the announced hostname.** `cloudflared`
   writes the line carrying its `*.trycloudflare.com` URL to **stderr**, not
   stdout (found by Task 8, after the plan was written). `_spawn` merges them
   with `stderr=STDOUT` so a manager watching stdout alone cannot wait out
   the whole timeout and then kill a perfectly healthy tunnel. No unit test
   catches this on its own, because a fake subprocess yields whatever the
   test feeds it -- so `tests/test_6b_transport_manager.py` runs a real
   `sys.executable` for that one assertion.

2. **A stop verifies that it stopped.** `tailscale serve --bg` and
   `tailscale funnel --bg` daemonise: the process TENKA spawned exits on its
   own, so terminating it again touches nothing, and only the argv
   `stop_command()` returns un-serves the mapping -- an argv that is
   documented but was never exercised against a live mapping, whose
   `<target>` grammar accepts free text, and which could therefore exit 0
   having served the literal string `off`. So the argv runs, and then the
   provider's own status is re-read and a `Web` entry still keyed under this
   transport's public port is a **loud failure**. An internet-facing listener
   that silently stayed up is the worst outcome this milestone can produce.
   The public port is recovered from the `--https` value of the adapter's own
   stop argv, because nothing on the `TransportAdapter` Protocol exposes it
   (`base.py`'s `stop_command` docstring says so at length) -- which also
   means the port being checked is read off the very command whose effect is
   being checked, so the two cannot drift apart. The *document* to check is
   named by the adapter's own `status_command`, never derived from the stop
   argv's verb -- deriving it assumed every provider's CLI is `<binary> <verb>
   status --json` and quietly asked a funnel-scoped view a serve-scoped
   question. A stop this module cannot verify -- no status command, an
   unreadable document, a shape it cannot sweep -- is reported UNVERIFIED,
   never as success: this check is the last thing standing, with no layer 3
   behind it.

   A stop or a shutdown arriving while a start is still in flight **cancels
   that start** rather than finding nothing and reporting success. The window
   is `HOSTNAME_TIMEOUT_SECONDS` wide, and what it costs is exactly the
   outcome above: a tunnel that comes up after the operator asked for
   everything to be down.

   The teardown itself, and the spawn it may need to recover a handle from,
   both survive being cancelled any number of further times while they are in
   flight (`_await_uncancellably`, fix round 3) -- not just once. anyio
   re-delivers a cancelled scope's cancellation on every event-loop cycle, so
   a route handler or a `stop_all` caller reaches a second and third repeat
   cancel within two loop cycles, and rounds 1 and 2 each survived only the
   first of those.

3. **Every synchronous provider call is wrapped in `asyncio.to_thread`.**
   `preflight()` and the status re-read are `subprocess.run` with a 10s
   timeout, and this manager runs on the event loop *the assistant herself
   shares*: a blocking call here stalls her, not just the API.

Layering: `io/api/` may import `core/` and `config` only, and this module
reaches at neither. `server.py` is imported **inside** the two functions that
need it, not at module level: `server` imports `app`, and `app` imports every
route module at import time, so a route module that imports this manager
(Task 13's `/v1/transports`) would close an import cycle. The deferred import
is also the seam the tests replace.
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import subprocess
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Iterator
from urllib.parse import urlparse

from ..listeners import local_port, port_for
from ..security import unpublish_host
from . import TransportRegistry, transport_registry
from .base import HOSTNAME_TIMEOUT_SECONDS, TransportAdapter, TransportSession

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..server import StudioListeners

logger = logging.getLogger(__name__)

__all__ = ["TransportError", "TransportManager", "is_serving", "require_serving"]

# The only hostnames a spawned argv may legitimately forward to. A tunnel
# target that is not one of these is not this daemon (every listener binds
# loopback, without exception -- `server.bind_listener` refuses anything else).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# How much provider output is scanned for the announced hostname before
# giving up. A chatty -- or hostile -- provider process must not be a memory
# story, and every provider announces its name in its first few lines or not
# at all. Given up on is the same outcome as never announced: the tunnel does
# not serve.
_MAX_SCAN_BYTES = 64 * 1024

# The `StreamReader` limit for one line of provider output. Separate from the
# total above so that a single enormous line fails the read rather than
# buffering until the total is reached.
_LINE_LIMIT_BYTES = 16 * 1024

# `subprocess.run` timeout for the two synchronous provider calls a teardown
# makes (the stop argv, and the status re-read that verifies it). Matches the
# adapters' own preflight timeout; both are local daemon queries.
_SYNC_TIMEOUT_SECONDS = 10.0

# How long a terminated provider process gets to exit before it is killed.
_REAP_TIMEOUT_SECONDS = 5.0


class TransportError(RuntimeError):
    """A transport could not be started, or could not be *proved* stopped.

    Carries a sentence fit to show an operator: the adapter's own refusal
    text, or a description of the misconfiguration this manager refused to
    act on. Task 13's route turns it into a 409, which is why it never
    contains a hostname, a token or a path.
    """


# ─── The invariant that is not in the type ───────────────────────────────────

def is_serving(session: TransportSession) -> bool:
    """Whether *session* is a tunnel that actually serves.

    `TransportSession.hostname` is `str | None` -- filled in after the
    announcement via `dataclasses.replace` -- so spec §4's "a tunnel that
    starts but never announces a hostname does not serve" cannot be enforced
    by the type. It is enforced here, and by `require_serving` below, and by
    `TransportManager.running()` filtering on it: a session with no hostname
    has no published name, so anything treating it as live would be offering
    a registered socket with no lock on the far side.
    """
    return session.hostname is not None


def require_serving(session: TransportSession) -> TransportSession:
    """*session*, or `TransportError` if it never announced a hostname."""
    if not is_serving(session):
        raise TransportError(
            f"transport '{session.policy_name}' has no announced hostname, so "
            f"it does not serve; refusing to treat it as running")
    return session


# ─── The impure edges, in one place each ─────────────────────────────────────

async def _spawn(argv: list[str]) -> asyncio.subprocess.Process:
    """Spawn *argv* with its two output streams merged.

    `stderr=asyncio.subprocess.STDOUT` is the load-bearing argument:
    `cloudflared` writes the banner carrying its `*.trycloudflare.com` URL to
    **stderr**, while the Tailscale adapters announce on stdout. A manager
    that watched one stream would wait out `HOSTNAME_TIMEOUT_SECONDS` for a
    name that had already arrived, tear the listener down, and kill a healthy
    tunnel. Merging is what lets this module not know or care which stream a
    given provider chose.

    `create_subprocess_exec`, never `shell=True`: the argv is a list built by
    the adapter from an integer and its own module constants, and handing it
    to a shell would put quoting between TENKA and the command she built
    deliberately. `stdin` is `DEVNULL` because no provider is ever asked
    anything -- a tunnel that wants a password should fail, not hang.
    """
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        limit=_LINE_LIMIT_BYTES,
    )


def _run_argv(argv: list[str]) -> tuple[int, str]:
    """Run *argv* to completion. **Blocks** -- every caller here wraps it in
    `asyncio.to_thread`, because this manager shares the assistant's own
    event loop.

    Returns `(returncode, stdout)`. A non-zero code is never treated as
    proof of anything on its own: `base.py`'s `stop_command` docstring is
    explicit that a stop's exit code is provisional and the provider's own
    status is the authority.
    """
    result = subprocess.run(argv, capture_output=True, text=True,
                            timeout=_SYNC_TIMEOUT_SECONDS, check=False)
    return result.returncode, result.stdout or ""


# ─── Argv reading (KI-17 layer 1, and the stop verification) ──────────────────

def _loopback_target_ports(argv: list[str]) -> set[int]:
    """Every loopback port *argv* forwards to, parsed out of the real vector.

    Read back out of the argv rather than trusted from the `port` the adapter
    was handed, because layer 1 is a check on the command line that will
    actually be executed -- an adapter that builds the wrong URL is exactly
    what it exists to catch. Non-loopback URLs are ignored (they are not this
    daemon), and so are tokens that are not URLs at all -- a bare `--https`
    value is a public port, not a target.
    """
    ports: set[int] = set()
    for token in argv:
        if not isinstance(token, str) or "://" not in token:
            continue
        try:
            parsed = urlparse(token)
            host = (parsed.hostname or "").strip("[]").lower()
            port = parsed.port
        except (AttributeError, TypeError, ValueError):
            continue
        if host in _LOOPBACK_HOSTS and port is not None:
            ports.add(int(port))
    return ports


def _public_port_from_argv(argv: list[str]) -> int | None:
    """The `--https` value in *argv*, or `None` if it names none.

    This is the public port a Tailscale mapping is *keyed under* -- the one
    thing a teardown must check and the one thing the `TransportAdapter`
    Protocol does not expose (`base.py`'s `stop_command` docstring). Both
    `--https 443` and `--https=443` are read, since the two spellings are
    interchangeable to the provider and only one of them is in the tree.
    """
    for index, token in enumerate(argv):
        if not isinstance(token, str):
            continue
        if token == "--https" and index + 1 < len(argv):
            candidate = argv[index + 1]
        elif token.startswith("--https="):
            candidate = token.split("=", 1)[1]
        else:
            continue
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


class _UnrecognisedStatus(Exception):
    """A provider status document this module cannot sweep.

    Raised rather than degraded-to-`False`, and that asymmetry with
    `tailscale.py`'s preflight -- which logs a warning and carries on -- is
    the whole point (fix round 1, Important 2). A preflight that degrades is
    contained by KI-17 layer 3, the per-listener `Host` gate; **this check has
    nothing behind it.** It is the last thing standing between a stop that
    silently failed and an internet-facing listener nobody knows is up, so a
    document it cannot read has to become an UNVERIFIED stop rather than a
    verified one.
    """


def _web_documents(payload: object) -> Iterator[dict]:
    """Every `Web` map in a provider status document.

    Both the top-level one (`--bg` mappings, which is what TENKA creates) and
    the ones nested under `Foreground`, because a foreground mapping keyed
    under our public port is an internet-facing listener whether or not it is
    ours -- and this function's caller is deciding whether a stop succeeded.

    Raises `_UnrecognisedStatus` when a key that should hold a map does not.
    An absent key is a different thing from a key of the wrong shape: absent
    means "no mappings of that kind", which is the ordinary clear case and the
    literal `{}` this machine prints when nothing is served; the wrong shape
    means this function is reading a document it does not understand, and it
    must not answer "nothing there" for one it never looked inside.
    """
    if not isinstance(payload, dict):
        raise _UnrecognisedStatus(
            f"the status document is a {type(payload).__name__}, not an object")
    web = payload.get("Web")
    if web is not None:
        if not isinstance(web, dict):
            raise _UnrecognisedStatus(
                f"its 'Web' value is a {type(web).__name__}, not an object")
        yield web
    foreground = payload.get("Foreground")
    if foreground is None:
        return
    if not isinstance(foreground, dict):
        raise _UnrecognisedStatus(
            f"its 'Foreground' value is a {type(foreground).__name__}, not an "
            f"object")
    for entry in foreground.values():
        if not isinstance(entry, dict):
            raise _UnrecognisedStatus(
                f"one of its 'Foreground' entries is a "
                f"{type(entry).__name__}, not an object")
        inner = entry.get("Web")
        if inner is None:
            continue
        if not isinstance(inner, dict):
            raise _UnrecognisedStatus(
                f"a 'Foreground' entry's 'Web' value is a "
                f"{type(inner).__name__}, not an object")
        yield inner


def _key_public_port(key: object) -> int | None:
    """The public-port half of a `Web` key (`"<hostname>:<port>"`).

    Never the hostname half: a refusal or a failure sentence out of this
    module names ports and commands, never a hostname, a token or a path.
    """
    if not isinstance(key, str) or ":" not in key:
        return None
    tail = key.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _keyed_under(payload: object, public_port: int) -> bool:
    """Whether any `Web` entry is still keyed under *public_port*.

    Deliberately "keyed under", not "targets": a mapping *targets* a local
    port and is *keyed under* a public one, so a check phrased the other way
    round would be trivially satisfied by every document and would report
    every stop as successful (`base.py` records that this is exactly the
    wording that leaked a Critical the first time it was written).

    Raises `_UnrecognisedStatus` for a document of an unreadable shape. A key
    whose trailing half is not a port number is logged and skipped instead of
    raising: an entry keyed in some shape this module does not know cannot be
    matched against a port either way, and the provider's own status is
    documented to key on `"<host>:<port>"`, so raising there would turn one
    odd entry into a permanent false UNVERIFIED on every stop. Logged, not
    silent -- that was the other half of the finding.
    """
    for web in _web_documents(payload):
        for key in web:
            port = _key_public_port(key)
            if port is None:
                logger.warning(
                    "[API] a provider status entry is keyed in a shape this "
                    "check does not recognise, so it could not be matched "
                    "against public port %d; skipped", public_port)
                continue
            if port == public_port:
                return True
    return False


def _a_child_may_exist(spawn: asyncio.Future | None) -> bool:
    """Whether a spawn may have produced a process nobody holds.

    Asked only when the manager has no handle, to decide whether the provider's
    own stop must still run (fix round 2, Minor 1). The three answers:

    - `None` -- the spawn was never reached (a layer-1 refusal). No child, and
      no mapping of TENKA's own to remove.
    - done and it raised -- the OS refused the exec (no binary, no permission).
      Same conclusion: nothing was started, so nothing must be un-served.
    - anything else -- still running, cancelled, or succeeded while the handle
      was lost. A child may exist, so the provider's stop runs. Unknown is
      answered as "yes" deliberately: the cost of an unnecessary `off` is one
      mapping this transport owns anyway, and the cost of skipping a necessary
      one is an internet-facing listener nobody knows about.
    """
    if spawn is None:
        return False
    if not spawn.done() or spawn.cancelled():
        return True
    return spawn.exception() is None


def _new_owner(name: str) -> str:
    """A fresh owner id for one *session* of a transport.

    Never the transport's name. Two successive runs of one tunnel get
    different hostnames from the provider, so they must be different owners
    -- otherwise stopping the second leaves the first's name trusted, which
    is the whole failure `PublishedHosts` exists to prevent. The name is a
    readable prefix only; the uuid is what makes it an identity.
    """
    return f"transport:{name}:{uuid.uuid4().hex}"


# ─── The manager ─────────────────────────────────────────────────────────────

class TransportManager:
    """Starts, stops and outlives nothing.

    Holds the `StudioListeners` handle `serve()` left behind, so a start can
    park its task and socket where a stop -- or a shutdown that never saw the
    start -- can find them again, and reaches `listener_policies`,
    `published_hosts` and `raises` through that handle's app state rather
    than holding copies: `HostGate` and `authenticate()` read the live dicts,
    so an entry added or dropped here is in force on the very next request.
    """

    def __init__(self, listeners: "StudioListeners", base_port: int, *,
                 registry: TransportRegistry | None = None) -> None:
        self._listeners = listeners
        self._base_port = int(base_port)
        self._registry = registry if registry is not None else transport_registry
        self._sessions: dict[str, TransportSession] = {}
        self._watchers: dict[str, asyncio.Task] = {}
        # Starts in flight, by name -- the task, not merely the name, because
        # a stop arriving mid-start has to be able to *cancel* it (fix round
        # 1, Important 1). Registered synchronously before the first `await`
        # in `start`, so two overlapping starts of one transport cannot both
        # pass the "already running" check and both bind.
        self._starting: dict[str, asyncio.Task] = {}
        # Names whose in-flight start was cancelled by `stop`, so `start` can
        # tell that cancellation apart from its own caller being cancelled and
        # answer with a `TransportError` instead of re-raising `CancelledError`
        # into a caller that was never cancelled.
        self._stopped_while_starting: set[str] = set()
        # What a start's unwind could not undo, by name. Read by whoever is
        # owed the news: the failing `start` itself, and a `stop` that
        # cancelled it. Cleared at the top of each start, so a read only ever
        # describes the most recent attempt on that name.
        self._unwind_failures: dict[str, list[str]] = {}

    # ─── Reading ─────────────────────────────────────────────────────────

    def running(self) -> dict[str, TransportSession]:
        """The transports that are serving right now, by policy name.

        Filtered on `is_serving`, not merely on membership: a session with no
        announced hostname never reaches this dict by the normal path, and
        filtering here means it could not be reported as running even if it
        did.
        """
        return {name: session for name, session in self._sessions.items()
                if is_serving(session)}

    def names(self) -> list[str]:
        """Every registered transport name, running or not -- what a listing
        route enumerates."""
        return self._registry.names()

    def adapter_for(self, name: str) -> TransportAdapter | None:
        return self._registry.get(name)

    # ─── Starting ────────────────────────────────────────────────────────

    async def start(self, name: str) -> TransportSession:
        """Start transport *name* and return its session.

        Raises `TransportError` on every refusal -- an unknown name, a
        transport already running, a preflight refusal (KI-17 layer 2), an
        argv that does not target this transport's own port (layer 1), or a
        tunnel that never announced a hostname. Every one of those unwinds
        whatever the start had already done before it raised.
        """
        if name in self._sessions:
            raise TransportError(f"transport '{name}' is already running")
        if name in self._starting:
            raise TransportError(f"transport '{name}' is already starting")
        adapter = self._registry.get(name)
        if adapter is None:
            raise TransportError(
                f"unknown transport '{name}' (registered: "
                f"{self._registry.names()})")
        try:
            port = port_for(name, self._base_port)
        except KeyError:
            raise TransportError(
                f"transport '{name}' has no port in the listener map, so "
                f"nothing declared where it would listen") from None
        if port == local_port(self._base_port):
            # Unreachable through the registry (which refuses the name
            # `local` outright) and checked anyway: a transport handed
            # local's port would inherit `POLICIES["local"]` in full.
            raise TransportError(
                f"transport '{name}' would bind port {port}, the loopback "
                f"listener's own port (KI-17); refusing to start")

        # Run the start as a task this manager owns, so a `stop` or a
        # `stop_all` arriving while it is in flight can cancel it. The
        # alternative -- refusing such a stop -- leaves the caller with a
        # tunnel that comes up *after* it asked for everything to be down,
        # and at shutdown that is a mapping still published to the open
        # internet after TENKA has exited.
        # Cleared *before* the task exists, not on its first line: a stop
        # landing in that one-tick window would otherwise cancel a task that
        # never ran and report the *previous* failed start's failures as its
        # own (fix round 2, Minor 2). Over-reporting is the safe direction, but
        # the message would be about a different attempt.
        self._unwind_failures.pop(name, None)
        task = asyncio.create_task(self._start_and_unwind(name, adapter, port),
                                   name=f"transport-start-{name}")
        self._starting[name] = task
        try:
            return await task
        except asyncio.CancelledError:
            # Whose cancellation was it? The marker is *not* evidence: a stop
            # may have set it while this caller's own task was independently
            # cancelled, and answering that caller with a `TransportError`
            # instead of a `CancelledError` is the shutdown-hang shape (fix
            # round 2, Minor 3). `cancelling()` is the only thing that knows:
            # it counts cancellation requests against *this* task, so it is
            # non-zero exactly when our caller was cancelled and zero when only
            # the inner start task was.
            outer = asyncio.current_task()
            if outer is not None and outer.cancelling() > 0:
                raise
            if name not in self._stopped_while_starting:
                # Nothing cancelled us on purpose; the start task was cancelled
                # from somewhere else and has already unwound. Re-raise, because
                # swallowing a cancellation is how a shutdown hangs.
                raise
            self._stopped_while_starting.discard(name)
            raise TransportError("; ".join([
                f"transport '{name}' was stopped while it was still starting",
                *self._unwind_failures.get(name, ()),
            ])) from None
        finally:
            self._starting.pop(name, None)
            # A marker left set would make the *next* start on this name read
            # its own caller's cancellation as a stop.
            self._stopped_while_starting.discard(name)

    async def _start_and_unwind(self, name: str, adapter: TransportAdapter,
                            port: int) -> TransportSession:
        # 1. Preflight, off the event loop -- the Tailscale adapters shell out
        # to `tailscale serve status --json` in here (spec §2.3 L2). Before
        # the bind, so a refusal has nothing to unwind.
        refusal = await asyncio.to_thread(adapter.preflight, port)
        if refusal:
            logger.warning("[API] transport '%s' refused to start: %s",
                           name, refusal)
            raise TransportError(refusal)

        # 2. Bind. From here on every failure runs the shared teardown.
        # Wrapped, not left to propagate: `_bind` raises a raw `OSError` on a
        # bind-collision race, and `errors.py` maps that to 404 -- while this
        # method's own docstring, and `routes/transports.py`'s, both promise
        # `TransportError` (409) for every refusal. Admin/loopback-only, so
        # the cost of leaving it raw was never a privilege crossing, only an
        # operator reading the wrong status code for an incident (fix round
        # 5, Important 3).
        try:
            sock = _bind(port)
        except OSError as exc:
            raise TransportError(
                f"transport '{name}' could not bind port {port} ({exc})"
            ) from exc
        owner = _new_owner(name)
        process: asyncio.subprocess.Process | None = None
        serve_task: asyncio.Task | None = None
        # The spawn task, once there is one. What gates the provider's `off`
        # argv is whether a **child may exist**, never whether one was
        # intended: a start refused by the layer-1 argv check, or a spawn the
        # OS refused outright (no binary, exec denied), created no mapping of
        # TENKA's own, and running `off` anyway would destroy whatever the
        # operator already had under that public port -- spec §2.3 L2's rule is
        # that a conflict is named, never silently repaired, and this is the
        # same principle on the way back out (fix round 1, Minor 4; narrowed
        # from "we were about to try" in fix round 2, Minor 1).
        spawn: asyncio.Future | None = None
        try:
            # 3. Register the port in the *live* registry, before the spawn:
            # the argv assertion below reads this dict, and a tunnel must
            # never be pointed at a port nothing has claimed.
            self._policies()[port] = name

            # 4. Spawn, with the argv checked against the registry first.
            argv = [str(token) for token in adapter.command(port)]
            self._refuse_a_foreign_target(argv, name=name, port=port)
            # `_await_uncancellably`, not a single shield-then-recover: a
            # cancel landing *inside* `create_subprocess_exec` would leave a
            # child this manager has no handle to -- an orphan tunnel, the
            # very thing a cancellable start is supposed to prevent -- and
            # round 2's single recovery survived exactly one repeat cancel
            # before a further one reached the spawn directly (fix round 3,
            # Important 2). A non-cancellation failure here (no such binary,
            # exec denied) is the OS refusing the exec, not a refusal this
            # manager issued, so it is wrapped to match this method's own
            # docstring -- `start` raises `TransportError` on every refusal.
            spawn = asyncio.ensure_future(_spawn(argv))
            try:
                process = await _await_uncancellably(spawn)
            except Exception as exc:
                raise TransportError(
                    f"transport '{name}' could not be spawned ({exc})"
                ) from exc
            # The spawn is done; only now is a cancellation delivered while
            # it was in flight safe to honour -- any earlier would race the
            # very child creation this exists to protect.
            _reraise_if_still_cancelling()
            logger.info("[API] transport '%s' spawned %s (pid %s) for port %d",
                        name, argv[0], getattr(process, "pid", "?"), port)

            # 5. The announced hostname, under a hard timeout. No name means
            # no serving: a registered socket with no published host is a
            # door with no lock on the far side (spec §4).
            hostname = await self._announced_hostname(adapter, process)
            if hostname is None:
                raise TransportError(
                    f"transport '{name}' never announced a hostname within "
                    f"{HOSTNAME_TIMEOUT_SECONDS:.0f}s, so it does not serve")

            # 6. Trust the name, scoped to this listener and this session --
            # and to this adapter's own public port, so `security.py`'s
            # `endpoint_origins`/`origins_for` can build the same
            # port-carrying URL a browser or a phone actually reaches this
            # transport at, without importing anything from `transports/`.
            self._publish(hostname, owner=owner, listener=port,
                          public_port=adapter.public_port())

            # 7. Serve, last: nothing answers on this socket until the name
            # it answers under is both known and published.
            serve_task = _serve(self._app(), sock, name=f"studio-{name}")

            session = require_serving(TransportSession(
                policy_name=name, port=port, owner=owner, process=process,
                sock=sock, serve_task=serve_task, hostname=hostname,
                adapter=adapter))
            self._sessions[name] = session
            self._listeners.tasks[name] = serve_task
            self._listeners.sockets[name] = sock
        except BaseException as exc:
            failures = await self._teardown_completely(
                adapter, policy_name=name, port=port, owner=owner,
                process=process, sock=sock, serve_task=serve_task,
                spawned=process is not None or _a_child_may_exist(spawn))
            for failure in failures:
                logger.error("[API] while unwinding transport '%s': %s",
                             name, failure)
            # Recorded for whoever is owed the news but cannot see the log:
            # `start`'s own caller (below) and a `stop` that cancelled us.
            self._unwind_failures[name] = failures
            # The teardown is done; only now is a cancellation delivered
            # while it was in flight safe to honour -- matching the other
            # two call sites (fix round 4). Without this, a direct cancel
            # landing here while `exc` is some unrelated failure (a natural
            # preflight or hostname-timeout refusal, say) would let this
            # task finish via that failure's exception while still carrying
            # an unhonoured `cancelling() > 0` -- recorded failures are
            # unaffected either way, since they are read from
            # `_unwind_failures`, not from which exception this task exits
            # through.
            _reraise_if_still_cancelling()
            if failures and isinstance(exc, TransportError):
                # The caller gets a 409 whose detail says both things: why the
                # start failed, and that the provider-side mapping could not
                # be proved gone (fix round 1, Minor 5). Losing the second half
                # to a log line is how an unverified stop becomes invisible.
                raise TransportError("; ".join([str(exc), *failures])) from exc
            raise

        self._supervise_output(name, adapter, session)
        logger.info("[API] transport '%s' is serving %s on port %d",
                    name, session.url, port)
        return session

    def _refuse_a_foreign_target(self, argv: list[str], *, name: str,
                                 port: int) -> None:
        """KI-17 layer 1, checked on the real argument vector before `Popen`.

        Three questions, cheapest and most dangerous first: is the target the
        loopback listener's own port, is it registered at all, and is it
        registered to *this* adapter's policy. A tunnel that answers any of
        them wrongly is a tunnel whose traffic would resolve to somebody
        else's ceiling -- which is KI-17 exactly.
        """
        targets = _loopback_target_ports(argv)
        if len(targets) != 1:
            raise TransportError(
                f"transport '{name}' built a command line with "
                f"{len(targets)} loopback targets ({sorted(targets)}); "
                f"exactly one is required to know what it would expose")
        target = next(iter(targets))
        local = local_port(self._base_port)
        if target == local:
            raise TransportError(
                f"transport '{name}' would point a tunnel at port {local}, "
                f"the loopback listener's own port -- every tunnelled "
                f"request would then resolve to the local policy (KI-17); "
                f"refusing to spawn")
        mapped = self._policies().get(target)
        if mapped is None:
            raise TransportError(
                f"transport '{name}' would point a tunnel at port {target}, "
                f"which no listener has registered; refusing to spawn")
        if mapped != name:
            raise TransportError(
                f"transport '{name}' would point a tunnel at port {target}, "
                f"which is registered to listener '{mapped}'; a tunnel must "
                f"only ever reach its own listener's ceiling; refusing to "
                f"spawn")
        if target != port:
            raise TransportError(
                f"transport '{name}' would point a tunnel at port {target}, "
                f"not the port {port} it was given; refusing to spawn")

    async def _announced_hostname(self, adapter: TransportAdapter,
                                  process: Any) -> str | None:
        """The hostname *process* announces, or `None` under any failure.

        `None` is one answer for four different things -- a timeout, an EOF
        (the process exited), a line longer than the reader's limit, and a
        scan that hit its byte bound -- because the consequence is the same
        for all of them: the tunnel does not serve.
        """
        try:
            return await asyncio.wait_for(self._scan(adapter, process),
                                          HOSTNAME_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning(
                "[API] transport '%s' announced no hostname in %.0fs",
                adapter.name, HOSTNAME_TIMEOUT_SECONDS)
            return None

    async def _scan(self, adapter: TransportAdapter,
                    process: Any) -> str | None:
        stream = getattr(process, "stdout", None)
        if stream is None:  # pragma: no cover - `_spawn` always pipes
            return None
        scanned = 0
        while scanned < _MAX_SCAN_BYTES:
            try:
                raw = await stream.readline()
            except (ValueError, asyncio.LimitOverrunError) as exc:
                logger.warning(
                    "[API] transport '%s' wrote a line past the reader's "
                    "limit (%s); giving up on its announcement",
                    adapter.name, exc)
                return None
            if not raw:
                logger.warning(
                    "[API] transport '%s' closed its output without "
                    "announcing a hostname", adapter.name)
                return None
            scanned += len(raw)
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            host = adapter.hostname_from(line)
            if host:
                return host
        logger.warning(
            "[API] transport '%s' wrote %d bytes without announcing a "
            "hostname; giving up rather than reading it forever",
            adapter.name, scanned)
        return None

    # ─── Stopping ────────────────────────────────────────────────────────

    async def stop(self, name: str) -> None:
        """Stop transport *name*, and prove it stopped.

        A name that is not running is not an error: a crash handler and an
        orderly shutdown both call this, and both may call it twice.

        A name that is *still starting* is neither. Reading only `_sessions`
        left a window up to `HOSTNAME_TIMEOUT_SECONDS` wide in which a stop
        found nothing, reported success, and the tunnel came up afterwards --
        at shutdown, a mapping published to the open internet after TENKA had
        exited, with nothing reporting anything (fix round 1, Important 1). So
        an in-flight start is cancelled and awaited, and its own unwind -- the
        `except BaseException` path that already handles `CancelledError` --
        is what tears down whatever it had allocated.

        Raises `TransportError` when the provider's own status still shows a
        mapping keyed under this transport's public port after its stop argv
        ran, or when that status could not be read at all -- everything else
        has been torn down by then, so the raise reports a failure rather than
        abandoning one.
        """
        starting = self._starting.get(name)
        if starting is not None and not starting.done():
            if name in self._stopped_while_starting:
                # Somebody already cancelled this start and is waiting for it
                # to finish unwinding. Measured on 3.11.9: a second cancel
                # lands on `asyncio.shield`'s own outer wrapper, not on the
                # teardown itself -- that is what a shield is for -- so it
                # takes a *third* cancel, arriving while round 2's recovery
                # re-await is unshielded, to reach the teardown directly and
                # abandon it. Round 3 removed that bound rather than naming
                # its next multiple: the wait below never cancels `starting`
                # a second time either way, and whatever teardown it may be
                # running is now protected for any number of further cancels,
                # not just one (see `_await_uncancellably`).
                logger.info(
                    "[API] transport '%s' is already being stopped mid-start; "
                    "waiting for that teardown rather than cancelling it",
                    name)
            else:
                logger.warning(
                    "[API] transport '%s' is still starting; cancelling its "
                    "start rather than letting it come up after this stop",
                    name)
                self._stopped_while_starting.add(name)
                starting.cancel()
            # `_wait_uncancellably`, never a plain `asyncio.wait` (fix round
            # 4): `asyncio.wait` never forwards a cancel onto `starting`
            # itself, however many times *this* wait is cancelled -- that
            # part was already right -- but it does not loop, so the first
            # such cancel propagates out of `stop` immediately, before
            # `starting`'s own (correctly protected) teardown has
            # necessarily finished. `stop_all`'s per-name handler then
            # records that teardown as "already completed" when it was not,
            # and nothing re-awaits the abandoned task afterward -- silent
            # at runtime, fatal at shutdown, which is `stop_all`'s entire
            # purpose. Waiting uncancellably here closes that gap the same
            # way the sibling branch below already closes it.
            await _wait_uncancellably(starting)
            # The starting transport's own teardown is done; only now is a
            # cancellation delivered while it was in flight safe to honour.
            _reraise_if_still_cancelling()
            failures = self._unwind_failures.get(name, [])
            if failures:
                raise TransportError("; ".join(failures))
            logger.info("[API] transport '%s' start cancelled and unwound",
                        name)
            return

        session = self._sessions.pop(name, None)
        if session is None:
            logger.debug("[API] transport '%s' is not running; nothing to stop",
                         name)
            return
        failures = await self._teardown_completely(
            session.adapter, policy_name=session.policy_name,
            port=session.port, owner=session.owner, process=session.process,
            sock=session.sock, serve_task=session.serve_task, spawned=True)
        # The teardown is done; only now is a cancellation delivered while it
        # was in flight safe to honour. Unlike the mid-start branch above,
        # this call was not entered by catching anything, so there is no
        # enclosing `raise` to carry a cancellation forward on its own --
        # it is checked and raised explicitly instead (fix round 3).
        if failures:
            logger.error(
                "[API] transport '%s' could not be fully stopped: %s",
                name, "; ".join(failures))
        _reraise_if_still_cancelling()
        if failures:
            raise TransportError("; ".join(failures))
        logger.info("[API] transport '%s' stopped", name)

    async def stop_all(self) -> list[str]:
        """Stop every running transport; return the failures, never raise.

        Each teardown is guarded individually so one transport that will not
        stop cannot strand the others -- and cannot strand the shutdown that
        called this, which still has the primary listener to cancel
        afterwards (`server.serve_socket`'s docstring: transports before the
        primary, always). The failures come back as strings so the caller can
        log or surface them; they are never swallowed.
        """
        failures: list[str] = []
        # Starting names included, and that is the half that matters: a
        # shutdown inside a start's 30-second window would otherwise leave the
        # tunnel it never saw to come up after the process was gone.
        for name in [*self._sessions, *self._starting]:
            try:
                await self.stop(name)
            except TransportError as exc:
                failures.append(str(exc))
                logger.error("[API] transport '%s' could not be stopped: %s",
                             name, exc)
            except asyncio.CancelledError:
                # `stop` only ever lets a cancellation reach here *after*
                # running its teardown to completion (fix round 3) -- so this
                # transport is not left half-stopped, only unreported -- and
                # `stop_all` must still attempt every other name rather than
                # abandon them to the same cancellation that produced this
                # one. Never re-raised: raising mid-shutdown would skip
                # main.py's cancel of the primary listener, which is worse
                # than a logged failure (round 1, concern 4) -- and that
                # concern is exactly as true of a cancellation as of any
                # other failure this loop already swallows.
                failures.append(
                    f"transport '{name}': its stop was cancelled after its "
                    f"teardown had already completed")
                logger.warning(
                    "[API] transport '%s' stop was cancelled (its teardown "
                    "had already completed)", name)
            except Exception as exc:  # pragma: no cover - defensive
                failures.append(f"transport '{name}' teardown raised: {exc}")
                logger.exception("[API] transport '%s' teardown raised", name)
        return failures

    async def _teardown_completely(self, adapter: TransportAdapter | None,
                                   **kwargs) -> list[str]:
        """`_teardown`, run to completion no matter how many times *this*
        task is cancelled while it is in flight.

        Found while testing Important 1's fix, and in the same family as it: a
        `stop` cancels an in-flight start, and if that start is already
        unwinding, the cancellation lands *inside* the teardown -- between the
        socket close and the provider's `off` argv, say. A teardown that stops
        halfway leaves exactly what the whole method exists to prevent, and
        the stop that caused it would have reported success.

        Round 1 and round 2 both shielded once and recovered with a single
        plain re-await, which survives exactly one repeat cancellation before
        a further one lands on the teardown task directly -- and anyio
        re-delivers a cancelled scope's cancellation on every event-loop
        cycle, so a route handler or a `stop_all` caller reaches that further
        cancel within two loop cycles, not some comfortable margin (fix round
        3). `_await_uncancellably` re-shields on every iteration instead of
        recovering once, so the teardown always runs to completion regardless
        of depth (probe-verified at 2, 3, 5 and 9).

        Returns the failures exactly as `_teardown` produced them, whether or
        not a cancellation arrived while waiting: a cancellation delivered
        here means the *caller* wants out, not that the failures stopped
        mattering, and a caller with bookkeeping to do about them (recording,
        deciding whether to raise) is better placed than this method to also
        decide when to honour that cancellation.
        """
        task = asyncio.ensure_future(self._teardown(adapter, **kwargs))
        return await _await_uncancellably(task)

    async def _teardown(self, adapter: TransportAdapter | None, *,
                        policy_name: str, port: int, owner: str,
                        process: Any | None, sock: socket.socket | None,
                        serve_task: asyncio.Task | None,
                        spawned: bool) -> list[str]:
        """The one teardown, in spec §4's reverse order. Idempotent.

        *spawned* says whether the provider was ever asked to do anything. It
        gates the provider-side stop: a start refused before its subprocess
        existed has no mapping of its own to remove, and running the `off`
        argv anyway would destroy whatever mapping the operator already had
        under that public port.

        Returns the failures that could not be undone -- an unverified
        provider stop, or a raise that could not be dropped. Both are things
        this method cannot fix by trying harder and must not report as
        success.
        """
        failures: list[str] = []

        # 1. The hostname stops being trusted first: while it is published it
        # is an accepted `Host` and a trusted `Origin`, and a name whose
        # tunnel is going away must stop being either before anything else.
        withdrawn = unpublish_host(self._app_state(), owner)
        if withdrawn:
            logger.info("[API] transport '%s' withdrew %d published name(s)",
                        policy_name, len(withdrawn))

        # 2. The port stops resolving to a policy, so a request that somehow
        # arrives on it is granted nothing at all. Guarded on the entry still
        # being ours: a second teardown must not drop a *later* session's.
        policies = self._policies()
        if policies.get(port) == policy_name:
            policies.pop(port, None)

        # 3. Raises scoped to this transport: one that outlived its listener
        # could never be exercised, and must not be inherited by the next
        # listener under the same name.
        raises = getattr(self._app_state(), "raises", None)
        if raises is not None:
            try:
                raises.drop_policy(policy_name)
            except Exception as exc:
                # Never swallowed (fix round 1, Minor 6): a raise that
                # survives its listener is inherited by the next session under
                # the same policy name, which is the exact inheritance
                # `drop_policy` exists to prevent.
                logger.error("[API] transport '%s': its ceiling raises could "
                             "not be dropped (%s)", policy_name, exc)
                failures.append(
                    f"transport '{policy_name}': its ceiling raises could not "
                    f"be dropped ({exc}) -- a raise scoped to this policy may "
                    f"be inherited by the next session under the same name")

        # 4. The crash watcher, before the things it watches go away.
        await _cancel(self._watchers.pop(policy_name, None))

        # 5. The listener itself, then its socket. `serve_socket` closes the
        # socket in a done-callback, and `socket.close()` is idempotent, so
        # closing it here too is belt and braces for the path where the task
        # was cancelled before it ever ran.
        await _cancel(serve_task)
        if self._listeners.tasks.get(policy_name) is serve_task:
            self._listeners.tasks.pop(policy_name, None)
        if self._listeners.sockets.get(policy_name) is sock:
            self._listeners.sockets.pop(policy_name, None)
        if sock is not None:
            with suppress(OSError):
                sock.close()

        # 6. The provider's own stop, and the verification that it worked --
        # only if the provider was ever asked for anything.
        if spawned:
            failures.extend(
                await self._provider_stop(adapter, policy_name, port))
        else:
            logger.debug(
                "[API] transport '%s' never reached its subprocess, so there "
                "is no provider-side mapping of its own to remove",
                policy_name)

        # 7. The subprocess. Last, because for a daemonising provider it has
        # already exited and for a foreground one this *is* the stop -- and
        # in neither case is it what un-serves the provider's mapping.
        failures.extend(await _reap(process, policy_name))
        return failures

    async def _provider_stop(self, adapter: TransportAdapter | None,
                             policy_name: str, port: int) -> list[str]:
        """Run `stop_command`'s argv when there is one, then verify it.

        `None` is a real answer, not an oversight: a foreground provider's
        spawned process *is* the tunnel, so reaping it (step 7) is the stop
        and there is nothing to verify. An argv means the provider
        daemonised, and then the exit code is provisional -- only the
        provider's own status, named by the same adapter through
        `status_command`, can say whether the mapping is gone.

        Every path out of here either verifies the stop or returns a sentence
        containing UNVERIFIED. There is no third answer, because the third
        answer is the failure this whole method exists to make impossible: a
        stop reported as successful when nothing checked.
        """
        if adapter is None:
            return [f"transport '{policy_name}': the session carries no "
                    f"adapter, so its provider-side stop could not be run or "
                    f"verified -- any mapping it left is UNVERIFIED and may "
                    f"still be up"]
        try:
            argv = adapter.stop_command(port)
        except Exception as exc:  # pragma: no cover - defensive
            return [f"transport '{policy_name}': its adapter could not name a "
                    f"stop command ({exc}), so the stop is UNVERIFIED"]
        if not argv:
            return []
        argv = [str(token) for token in argv]

        try:
            code, _out = await asyncio.to_thread(_run_argv, argv)
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"transport '{policy_name}': its stop command "
                    f"{argv[0]} could not be run ({exc}), so the stop is "
                    f"UNVERIFIED -- an internet-facing listener may still be up"]
        if code != 0:
            logger.warning(
                "[API] transport '%s' stop command exited %d; the status "
                "re-read below is what decides", policy_name, code)

        # The public port comes off the very command whose effect is being
        # checked (`base.py`'s `stop_command` docstring requires exactly this,
        # and it is why the two cannot drift apart). The *reader* comes from
        # the adapter instead of being derived from that argv's verb: an
        # adapter has already decided which of its provider's documents it
        # trusts, and guessing a second one behind its back is how a
        # funnel-scoped view came to be used for a serve-scoped question.
        public_port = _public_port_from_argv(argv)
        if public_port is None:
            return [f"transport '{policy_name}': its stop command names no "
                    f"public port to check, so the stop is UNVERIFIED -- an "
                    f"internet-facing listener may still be up"]
        try:
            status_argv = adapter.status_command(port)
        except Exception as exc:  # pragma: no cover - defensive
            status_argv = None
            logger.warning("[API] transport '%s' could not name a status "
                           "command (%s)", policy_name, exc)
        if not status_argv:
            return [f"transport '{policy_name}': its adapter names a stop "
                    f"command but no status command, so the stop of public "
                    f"port {public_port} is UNVERIFIED -- an internet-facing "
                    f"listener may still be up"]
        status_argv = [str(token) for token in status_argv]
        try:
            _code, out = await asyncio.to_thread(_run_argv, status_argv)
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"transport '{policy_name}': its status could not be "
                    f"re-read ({exc}), so the stop of public port "
                    f"{public_port} is UNVERIFIED"]
        try:
            payload = json.loads(out)
        except ValueError as exc:
            return [f"transport '{policy_name}': its status was not readable "
                    f"JSON ({exc}), so the stop of public port {public_port} "
                    f"is UNVERIFIED -- an internet-facing listener may still "
                    f"be up"]
        try:
            still_there = _keyed_under(payload, public_port)
        except _UnrecognisedStatus as exc:
            return [f"transport '{policy_name}': its status could not be "
                    f"swept ({exc}), so the stop of public port "
                    f"{public_port} is UNVERIFIED -- an internet-facing "
                    f"listener may still be up"]
        if still_there:
            return [f"transport '{policy_name}': a mapping is STILL keyed "
                    f"under public port {public_port} after its stop command "
                    f"ran -- an internet-facing listener may still be up; "
                    f"check the provider's own status and remove that one "
                    f"mapping by hand"]
        logger.info("[API] transport '%s' verified: no mapping remains under "
                    "public port %d", policy_name, public_port)
        return []

    # ─── The crash path ──────────────────────────────────────────────────

    def _supervise_output(self, name: str, adapter: TransportAdapter,
                          session: TransportSession) -> None:
        """Drain every provider's output; tear down only when an exit means
        the tunnel died.

        Which exits mean that is read off the Protocol, never off which
        transport this is: `stop_command()` returning `None` means the
        spawned process's own lifetime *is* the tunnel's, so its exit is a
        crash. A provider that returns an argv daemonised -- its spawned
        process exits immediately and by design, and treating that as a crash
        would tear the tunnel down seconds after starting it.

        The drain runs either way, because a pipe nobody reads is a hazard
        regardless of who owns the tunnel's lifetime.
        """
        try:
            daemonises = adapter.stop_command(session.port) is not None
        except Exception:  # pragma: no cover - defensive
            daemonises = True
        self._watchers[name] = asyncio.create_task(
            self._supervise(name, session, teardown_on_exit=not daemonises),
            name=f"transport-watch-{name}")

    async def _supervise(self, name: str, session: TransportSession, *,
                         teardown_on_exit: bool) -> None:
        """Drain the provider's output, and tear down if its exit killed the
        tunnel.

        The drain is not optional bookkeeping: a foreground provider writes
        to a pipe for its whole life, and a pipe nobody reads fills and
        blocks the process that is writing to it. So this task reads and
        discards until EOF, and EOF is also how it learns the process is on
        its way out.
        """
        process = session.process
        stream = getattr(process, "stdout", None)
        if stream is not None:
            while True:
                try:
                    chunk = await stream.read(_LINE_LIMIT_BYTES)
                except Exception:  # pragma: no cover - stream torn down
                    break
                if not chunk:
                    break
        with suppress(Exception):
            await process.wait()
        if not teardown_on_exit:
            logger.debug(
                "[API] transport '%s' spawn process exited (code %s); its "
                "provider daemonises, so the tunnel is unaffected",
                name, getattr(process, "returncode", None))
            return

        current = self._sessions.get(name)
        if current is None or current.owner != session.owner:
            return  # an orderly stop already took this session
        logger.error(
            "[API] transport '%s' exited on its own (code %s); tearing its "
            "listener down so nothing outlives it",
            name, getattr(process, "returncode", None))
        # Deregister *this* watcher before tearing down. The teardown cancels
        # the session's watcher, and it now runs in a task of its own
        # (`_teardown_completely`), so `asyncio.current_task()` inside it is no
        # longer this task -- meaning it would cancel the very task that is
        # awaiting it, and the two would wait on each other forever. Dropping
        # the registration first leaves the teardown nothing to cancel.
        if self._watchers.get(name) is asyncio.current_task():
            self._watchers.pop(name, None)
        try:
            await self.stop(name)
        except TransportError as exc:
            logger.error("[API] transport '%s' crashed and its stop could "
                         "not be verified: %s", name, exc)

    # ─── App state, reached live rather than copied ───────────────────────

    def _app(self) -> Any:
        return self._listeners.app

    def _app_state(self) -> Any:
        return self._listeners.app.state

    def _policies(self) -> dict[int, str]:
        """The live `listener_policies` dict -- never a copy. `HostGate` and
        `authenticate()` hold this same object, so an entry written here is
        in force on the next request and an entry dropped here grants
        nothing from that moment on."""
        return self._app_state().listener_policies

    def _publish(self, hostname: str, *, owner: str, listener: int,
                public_port: int) -> None:
        published = getattr(self._app_state(), "published_hosts", None)
        if published is None:  # pragma: no cover - create_app always sets one
            raise TransportError(
                "this app has no published-hosts collection, so a tunnel's "
                "hostname could not be trusted; refusing to serve it")
        published.publish(hostname, owner=owner, listener=listener,
                          public_port=public_port)


# ─── Deferred `server` reach (see the module docstring) ───────────────────────

def _bind(port: int) -> socket.socket:
    """`server.bind_listener(port)`, imported at call time.

    Deferred because `server` imports `app`, and `app` imports every route
    module at import time -- so a route module importing this manager would
    close a cycle if this import sat at module level. It is also the seam the
    tests replace, which is why the lookup happens on the module rather than
    on a name bound once at import.
    """
    from .. import server
    return server.bind_listener(port)


def _serve(app: Any, sock: socket.socket, *, name: str) -> asyncio.Task:
    """`server.serve_socket(app, sock, name=name)` -- never `primary=True`.

    A transport listener must not own the app's lifespan or the process's
    signal handlers: one `EventHub` is shared by every listener, and the
    first primary to stop stops it for all of them
    (`server.serve_socket`'s docstring).
    """
    from .. import server
    return server.serve_socket(app, sock, name=name, primary=False)


# ─── Small async utilities ───────────────────────────────────────────────────

async def _settle_uncancellably(fut: "asyncio.Future") -> None:
    """Re-shield *fut* until it is actually done, no matter how many times
    the caller's own task is cancelled while waiting.

    A plain `await fut` forwards every cancellation of the *caller* onto
    *fut* (`Task.cancel()` cancels whatever future the task is currently
    suspended on); one `asyncio.shield` only defers exactly one such cancel,
    landing it on the shield's own outer wrapper instead of *fut*. Rounds 1
    and 2 both recovered from that one cancel with a single plain re-await,
    which is itself unshielded -- so a *further* cancel, arriving while that
    recovery is in flight, reaches *fut* directly and abandons it. anyio
    re-delivers a cancelled scope's cancellation on every event-loop cycle
    (`anyio/_backends/_asyncio.py`, reschedule at line 623), so a route
    handler or a `stop_all` caller reaches that further cancel within two
    loop cycles -- not some comfortable margin an enumeration of callers
    could hope to bound.

    Re-shielding on *every* iteration, rather than recovering once, removes
    the bound entirely: probe-verified robust at cancellation depths 2, 3, 5
    and 9 (fix round 3). Bounded by *fut* itself, not by anything added
    here -- every caller wraps something whose own steps are already bounded
    (subprocess timeouts, `HOSTNAME_TIMEOUT_SECONDS`), so a fresh timeout
    here would only duplicate that, not add safety.

    The shared body behind `_await_uncancellably` and `_wait_uncancellably`
    below -- round 4 found a *third* call site (`stop`'s "still starting"
    branch) that needed the same protection but not the same return shape,
    which is what split this out rather than adding a third near-duplicate.
    """
    while not fut.done():
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            continue


async def _await_uncancellably(fut: "asyncio.Future") -> Any:
    """`_settle_uncancellably`, then hand back *fut*'s own result or
    exception untouched.

    Deliberately does not re-raise a cancellation of the *caller* itself
    once *fut* is done -- the caller's own task still carries the
    outstanding request (`asyncio.current_task().cancelling()`), and callers
    with bookkeeping to do about *fut*'s result (recording failures, say)
    are better placed than this generic wait to decide when to also honour
    it (see `_reraise_if_still_cancelling`).
    """
    await _settle_uncancellably(fut)
    return fut.result()


async def _wait_uncancellably(fut: "asyncio.Future") -> None:
    """`_settle_uncancellably`, for callers that only need to know *fut* is
    done -- never its result or exception.

    `stop`'s "still starting" branch waits on the in-flight start's own
    task, which it (or a concurrent stop) may itself have cancelled, so
    *fut* routinely ends up cancelled or raising on its own account; that
    outcome is not this caller's to propagate -- `_unwind_failures` is the
    channel that branch reports through instead (matching what plain
    `asyncio.wait` used to leave alone before round 4 replaced it with
    this). *fut*'s own exception, if it has one and was not a cancellation,
    is still retrieved and discarded here, so asyncio does not log an
    "exception was never retrieved" warning for a future nobody else reads.
    """
    await _settle_uncancellably(fut)
    if not fut.cancelled():
        fut.exception()


def _reraise_if_still_cancelling() -> None:
    """Raise a fresh `CancelledError` if the current task still carries an
    outstanding cancellation request.

    Called only *after* whatever must not be interrupted -- a teardown, a
    spawn -- has already finished, so a cancellation delivered while that was
    in flight is honoured now rather than lost, and never a moment earlier:
    checking before completion would honour it in place of the very
    completion `_await_uncancellably` exists to guarantee (fix round 3).
    """
    current = asyncio.current_task()
    if current is not None and current.cancelling() > 0:
        raise asyncio.CancelledError()


async def _cancel(task: asyncio.Task | None) -> None:
    """Cancel *task* and wait for it to finish unwinding.

    Waiting matters: `serve_socket`'s cancellation handler is what runs
    uvicorn's own shutdown and frees the port, so a teardown that cancelled
    without awaiting would return with the socket still bound.

    Skips the task it is running inside, which is now a **belt-and-braces
    check that protects nothing on its own** (fix round 2, Minor 4). It used to
    be what stopped a crash watcher from cancelling itself mid-teardown, but
    the teardown runs in its own task (`_teardown_completely`) and
    `current_task()` inside it is that task, not the watcher. The real
    protection is `_supervise` **deregistering itself from `_watchers` before
    calling `stop`** -- do not delete that as redundant on the strength of this
    guard, or the teardown will cancel the task awaiting it and the two will
    wait on each other forever.
    """
    if task is None or task is asyncio.current_task():
        return
    task.cancel()
    with suppress(Exception):
        await asyncio.gather(task, return_exceptions=True)


async def _reap(process: Any | None, policy_name: str) -> list[str]:
    """Terminate *process* if it is still alive, then wait for it; return the
    failure, as the same UNVERIFIED-sentence list every other teardown step
    uses, if its death could not be confirmed.

    Not the stop for a daemonising provider (`_provider_stop` is), and the
    whole stop for a foreground one -- so for that provider this call *is*
    the mapping's teardown, and a process this cannot confirm dead is a
    mapping this cannot confirm gone. Escalates to `kill()` because a
    provider that ignores a terminate must not keep a tunnel open past this
    call; if `kill()` also fails to produce a confirmed exit, that is
    reported rather than swallowed (fix round 5, Critical 1 -- a prior
    version returned normally here, so `stop()` claimed success while a
    foreground provider's process, and its live edge connection, kept
    running).
    """
    if process is None or getattr(process, "returncode", None) is not None:
        return []
    with suppress(OSError, ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), _REAP_TIMEOUT_SECONDS)
        return []
    except Exception:
        # `asyncio.TimeoutError` is the expected one; anything else out of a
        # provider's `wait()` earns the same escalation rather than leaving a
        # process this manager promised to have reaped.
        logger.warning("[API] a transport subprocess ignored terminate; "
                       "killing it")
    with suppress(OSError, ProcessLookupError):
        process.kill()
    try:
        await asyncio.wait_for(process.wait(), _REAP_TIMEOUT_SECONDS)
        return []
    except Exception as exc:
        logger.error(
            "[API] transport '%s' subprocess could not be confirmed dead "
            "after kill() (%s); its mapping is UNVERIFIED and may still be "
            "up", policy_name, exc)
        return [f"transport '{policy_name}': its subprocess could not be "
                f"confirmed dead after kill() ({exc}) -- the mapping is "
                f"UNVERIFIED and may still be up"]
