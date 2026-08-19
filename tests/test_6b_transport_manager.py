"""Milestone 6b Task 9 -- `TransportManager`, the transport lifecycle.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.3, §4.

**No test in this file spawns a real tunnel.** `cloudflared` is not installed
on this machine and `tailscale` is -- running either would mutate the
operator's own state -- so every provider subprocess is a scripted fake, and
the two Tailscale adapters' status read is neutralised at
`tailscale._run_serve_status` for the tests that use the real adapters. The
one test that runs a *real* subprocess runs `sys.executable -c`, deliberately:
it is the only way to prove the manager reads a hostname announced on
**stderr**, which is where `cloudflared` writes its banner (Task 8's finding),
and a fake that yields whatever the test feeds it can never prove that.

Test doubles, planned up front per controller Ruling 13(b): `TransportSession`
carries `process`, `sock` and `serve_task`, so constructing one needs a fake
process (`_FakeProcess`), a real socket (from the real `bind_listener` -- a
closed-socket assertion is only worth making against a real file descriptor)
and a real `asyncio.Task`.
"""
from __future__ import annotations

import asyncio
import json
import re
import socket
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from assistant.core.capabilities import Capability
from assistant.io.api import server as server_mod
from assistant.io.api.listeners import local_port, port_for
from assistant.io.api.raises import RaiseStore
from assistant.io.api.security import PublishedHosts
from assistant.io.api.server import StudioListeners
from assistant.io.api.transports import TransportRegistry
from assistant.io.api.transports import manager as manager_mod
from assistant.io.api.transports import tailscale as tailscale_mod
from assistant.io.api.transports.base import TransportSession
from assistant.io.api.transports.manager import (
    TransportError,
    TransportManager,
    is_serving,
    require_serving,
)

# Not `config.STUDIO_API_PORT`: a test must never bind the port a real daemon
# on this machine may be holding, and the KI-17 assertions need a base whose
# whole block of four ports is the test's own.
BASE = 18787

_LOOPBACK_TARGET_RE = re.compile(r"http://127\.0\.0\.1:(\d+)")

# A sentinel, so `status_argv=None` can mean 'names no reader' rather than
# 'take the default'.
_UNSET: object = object()


# ═════════════════════════════════════════════════════════════════════════
# Doubles
# ═════════════════════════════════════════════════════════════════════════

class _FakeStream:
    """A scripted `StreamReader`: the lines a provider process 'wrote'.

    `reads` is what a boundedness assertion needs -- a manager that scanned
    forever would consume every line, and only counting the reads can tell
    the difference between 'gave up' and 'ran out of input'.
    """

    def __init__(self, lines: list[bytes], *, stall: bool = False) -> None:
        self._lines = list(lines)
        self._stall = stall
        self.reads = 0

    async def readline(self) -> bytes:
        self.reads += 1
        if self._lines:
            return self._lines.pop(0)
        if self._stall:
            await asyncio.sleep(3600)
        return b""

    async def read(self, n: int = -1) -> bytes:
        self.reads += 1
        if self._lines:
            return self._lines.pop(0)
        if self._stall:
            await asyncio.sleep(3600)
        return b""


class _FakeProcess:
    """A duck-typed `asyncio.subprocess.Process`."""

    def __init__(self, lines: list[bytes], *, stall: bool = False) -> None:
        self.stdout = _FakeStream(lines, stall=stall)
        self.pid = 4242
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        if self.returncode is None:
            await self._exited.wait()
        return self.returncode  # type: ignore[return-value]

    def terminate(self) -> None:
        self.terminated += 1
        self.exit(-15)

    def kill(self) -> None:
        self.killed += 1
        self.exit(-9)

    def exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
        self._exited.set()


class _FakeAdapter:
    """A duck-typed `TransportAdapter`, with every impure thing scripted.

    `target_port` is the lie a KI-17 layer-1 test needs: an adapter whose
    `command()` points somewhere other than the port it was handed.
    """

    def __init__(self, name: str, *, events: list[str] | None = None,
                 refusal: str | None = None, target_port: int | None = None,
                 stop_argv: list[str] | None = None,
                 status_argv: list[str] | None = _UNSET,
                 public_port: int = 443) -> None:
        self.name = name
        self._events = events if events is not None else []
        self.refusal = refusal
        self._target_port = target_port
        self._stop_argv = stop_argv
        self._public_port = public_port
        # Defaults to the provider's own reader, in the shape the Tailscale
        # adapters use. `None` is passed explicitly by the test that asserts a
        # stop it cannot verify.
        self._status_argv = ([f"fake-{name}", "serve", "status", "--json"]
                             if status_argv is _UNSET else status_argv)
        self.preflight_calls: list[int] = []
        self.stop_calls: list[int] = []

    def command(self, port: int) -> list[str]:
        target = self._target_port if self._target_port is not None else int(port)
        return [f"fake-{self.name}", "--https", "9443",
                f"http://127.0.0.1:{target}"]

    def hostname_from(self, line: str) -> str | None:
        text = line.strip()
        if not text.startswith("https://"):
            return None
        host = text[len("https://"):].split("/")[0]
        return host if host.endswith(".example.com") else None

    def public_port(self) -> int:
        """Defaults to 443 -- distinct from `command()`'s literal fake
        `9443` on purpose, so every existing test in this file (none of
        which asserts a ported URL) stays on the plain, no-port branch of
        `public_url` below. Overridable per instance
        (`test_start_publishes_the_adapters_own_public_port` is the one test
        that does) so this harness can prove `TransportManager._publish`
        actually forwards *this* value into `PublishedHosts`, rather than a
        hardcoded one, without duplicating the real adapters' own
        port-in-URL behaviour -- that stays pinned in
        `test_6b_transport_adapters.py`."""
        return self._public_port

    def public_url(self, hostname: str) -> str:
        """Bare `https://{hostname}` -- this harness's own fake public port
        (`command()`'s literal `9443`) plays no part in what `session.url`
        is asserted to be anywhere in this file; the real adapters' own
        port-in-URL behaviour is pinned in `test_6b_transport_adapters.py`
        instead."""
        return f"https://{hostname}"

    def preflight(self, port: int) -> str | None:
        self._events.append("preflight")
        self.preflight_calls.append(int(port))
        return self.refusal

    def stop_command(self, port: int) -> list[str] | None:
        self.stop_calls.append(int(port))
        return list(self._stop_argv) if self._stop_argv else None

    def status_command(self, port: int) -> list[str] | None:
        return list(self._status_argv) if self._status_argv else None


class _RecordingPolicies(dict):
    """`app.state.listener_policies`, with its writes observable.

    Ordering is the whole assertion of the first test in this file, and a
    plain dict cannot say when it was written to.
    """

    def __init__(self, initial: dict, events: list[str]) -> None:
        super().__init__(initial)
        self._events = events

    def __setitem__(self, key, value) -> None:
        self._events.append("register")
        super().__setitem__(key, value)

    def pop(self, key, *default):
        self._events.append("deregister")
        return super().pop(key, *default)


class _RecordingHosts(PublishedHosts):
    """The real `PublishedHosts` -- real behaviour, observable calls."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def publish(self, hostname: str, *, owner: str, listener: int,
                public_port: int | None = None) -> None:
        self._events.append("publish")
        super().publish(hostname, owner=owner, listener=listener,
                        public_port=public_port)

    def unpublish(self, owner: str):
        self._events.append("unpublish")
        return super().unpublish(owner)


class _Harness:
    """One manager, its app state, and every seam it reaches through."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.spawned: list[list[str]] = []
        self.served: list[str] = []
        self.ran: list[list[str]] = []
        self.processes: list[_FakeProcess] = []
        self.sockets: list[socket.socket] = []
        self.serve_tasks: list[asyncio.Task] = []
        self.lines_for: dict[str, list[bytes]] = {}
        self.stall: set[str] = set()
        self.status_payload: object = {}
        self.registry = TransportRegistry()
        self.policies = _RecordingPolicies({BASE: "local"}, self.events)
        self.published = _RecordingHosts(self.events)
        self.raises = RaiseStore()
        state = SimpleNamespace(listener_policies=self.policies,
                                published_hosts=self.published,
                                raises=self.raises)
        self.app = SimpleNamespace(state=state)
        self.local_sock = socket.socket()
        self.listeners = StudioListeners(
            app=self.app,  # type: ignore[arg-type]
            tasks={"local": _LocalTaskStandIn()},  # type: ignore[dict-item]
            sockets={"local": self.local_sock},  # type: ignore[dict-item]
        )
        self.manager = TransportManager(self.listeners, BASE,
                                        registry=self.registry)

    def register(self, adapter, lines: list[bytes] | None = None) -> None:
        self.registry.register(adapter.name, adapter)
        if lines is None:
            lines = [b"starting up\n",
                     f"https://{adapter.name}.example.com\n".encode()]
        self.lines_for[adapter.name] = list(lines)

    def close(self) -> None:
        for task in self.serve_tasks:
            task.cancel()
        for sock in self.sockets + [self.local_sock]:
            try:
                sock.close()
            except OSError:
                pass


class _LocalTaskStandIn:
    """`StudioListeners.tasks['local']`. The manager must never touch the
    local listener's entry, so this is deliberately not a real task: anything
    that cancels or awaits it fails loudly instead of quietly succeeding."""

    def done(self) -> bool:
        return False


def _sleeping_task(name: str) -> asyncio.Task:
    return asyncio.get_running_loop().create_task(
        asyncio.sleep(3600), name=f"fake-serve-{name}")


def _name_from_argv(argv: list[str]) -> str | None:
    """Which transport this argv belongs to, read off its loopback target.

    Works for the fakes and for the three real adapters alike, so a test may
    swap one for the other without the harness noticing.
    """
    for token in argv:
        match = _LOOPBACK_TARGET_RE.search(token)
        if match is None:
            continue
        from assistant.io.api.listeners import policy_name_for_port
        return policy_name_for_port(int(match.group(1)), BASE)
    return None


@pytest.fixture
def h(monkeypatch):
    harness = _Harness()

    real_bind = server_mod.bind_listener
    # Captured before the patch below, for the one test that needs the real
    # subprocess machinery back (`..._announced_only_on_stderr_is_read`).
    harness.real_spawn = manager_mod._spawn

    def _bind(port, host="127.0.0.1"):
        harness.events.append("bind")
        sock = real_bind(port, host)
        harness.sockets.append(sock)
        return sock

    async def _spawn(argv):
        harness.events.append("spawn")
        harness.spawned.append(list(argv))
        name = _name_from_argv(argv) or ""
        proc = _FakeProcess(harness.lines_for.get(name, []),
                            stall=name in harness.stall)
        harness.processes.append(proc)
        return proc

    def _serve(app, sock, *, name, primary=False):
        harness.events.append("serve")
        harness.served.append(name)
        task = _sleeping_task(name)
        harness.serve_tasks.append(task)
        return task

    def _run(argv):
        harness.ran.append(list(argv))
        # Any argv asking for JSON is a status read -- keyed on `--json`
        # rather than on the token "status", so a test may give an adapter a
        # reader that does not spell it that way (the whole point of
        # `status_command` being the adapter's to name).
        if "--json" in argv:
            return 0, json.dumps(harness.status_payload)
        return 0, ""

    # Kept on the harness so a test that replaces `_spawn` for its own
    # purposes can put this one back (the vacuity half of the refused-spawn
    # test does exactly that).
    harness.fake_spawn = _spawn

    monkeypatch.setattr(server_mod, "bind_listener", _bind)
    monkeypatch.setattr(server_mod, "serve_socket", _serve)
    monkeypatch.setattr(manager_mod, "_spawn", _spawn)
    monkeypatch.setattr(manager_mod, "_run_argv", _run)
    yield harness
    harness.close()


def _use_real_adapters(h, monkeypatch) -> None:
    """The three shipped adapters, with their one impure read neutralised.

    `tailscale._run_serve_status` is what `preflight` blocks on; replaced with
    a clear document so no test in this file runs the `tailscale` binary the
    operator's machine actually has installed.
    """
    monkeypatch.setattr(tailscale_mod, "_run_serve_status", lambda: {})
    h.registry.register("tailnet", tailscale_mod.TailnetAdapter())
    h.registry.register("funnel", tailscale_mod.FunnelAdapter())
    h.lines_for["tailnet"] = [b"Available within your tailnet:\n",
                              b"https://box.tail1234.ts.net/\n"]
    h.lines_for["funnel"] = [b"Funnel on:\n",
                             b"https://box.tail1234.ts.net/\n"]


async def _settle(predicate, *, tries: int = 200) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


def _effects(h, name: str, port: int, sock, proc, task, device: str) -> dict:
    """Everything a teardown is supposed to have done, in one comparable
    value -- so 'a crash runs the same teardown as a stop' can be asserted
    as an equality rather than as two lists of assertions that might drift."""
    return {
        "published": h.published.hosts_for(port),
        "owners": h.published.owners(),
        "registered": port in h.policies,
        "parked_task": name in h.listeners.tasks,
        "parked_socket": name in h.listeners.sockets,
        "raise": h.raises.capabilities_for(device, name),
        "socket_closed": sock.fileno() == -1,
        "serve_task_done": task.done(),
        "process_reaped": proc.returncode is not None,
        "running": sorted(h.manager.running()),
    }


# ═════════════════════════════════════════════════════════════════════════
# Start
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_start_binds_registers_spawns_and_publishes_in_that_order(h):
    """Spec §4's start sequence, asserted as a sequence.

    Each step is undoable only if the previous ones already happened, so the
    order is the contract and not an implementation detail: a spawn before
    the registration is a tunnel pointed at a port nothing has claimed, and a
    publish before the hostname is a name nobody announced.
    """
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)

    session = await h.manager.start("funnel")

    assert h.events == ["preflight", "bind", "register", "spawn", "publish",
                        "serve"], h.events
    assert session.policy_name == "funnel"
    assert session.port == port_for("funnel", BASE)
    assert session.hostname == "funnel.example.com"
    assert session.url == "https://funnel.example.com"
    assert adapter.preflight_calls == [port_for("funnel", BASE)]
    assert h.policies[session.port] == "funnel"
    assert h.published.hosts_for(session.port) == frozenset(
        {"funnel.example.com"})
    assert h.served == ["studio-funnel"]
    assert sorted(h.manager.running()) == ["funnel"]
    assert h.listeners.tasks["funnel"] is session.serve_task
    assert h.listeners.sockets["funnel"] is session.sock


@pytest.mark.asyncio
async def test_start_publishes_the_adapters_own_public_port(h):
    """`_publish` must forward *this adapter's* `public_port()`, not a
    hardcoded value -- the live-test defect's actual wiring, one layer below
    `TransportSession.url` (which reads the adapter directly and so cannot
    catch a broken `_publish` call on its own): `security.py`'s
    `endpoint_origins`/`origins_for` and `routes/pairing.py`'s `_endpoints()`
    both read the port back out of `PublishedHosts`, never the adapter, so
    if `start()` published the wrong port -- or none at all -- those two
    would silently build an unreachable origin/URL even though
    `session.url` itself was correct.
    """
    adapter = _FakeAdapter("tailnet", events=h.events, public_port=8443)
    h.register(adapter)

    session = await h.manager.start("tailnet")

    assert h.published.hosts_for(session.port) == frozenset(
        {"tailnet.example.com"})
    assert h.published.origins_for(session.port) == frozenset(
        {"https://tailnet.example.com:8443"})


@pytest.mark.asyncio
async def test_a_failed_preflight_stops_before_anything_is_bound(h):
    """KI-17 layer 2 is only a defence if it runs *first*.

    The second half of this test is the vacuity guard: the same adapter, the
    same harness, one field changed -- so a manager that never bound anything
    at all could not pass it.
    """
    adapter = _FakeAdapter("funnel", events=h.events,
                           refusal="tailscale serve already forwards port 443 "
                                   "straight to port 18787")
    h.register(adapter)

    with pytest.raises(TransportError) as excinfo:
        await h.manager.start("funnel")

    assert "18787" in str(excinfo.value)
    assert h.events == ["preflight"], h.events
    assert h.spawned == []
    assert h.sockets == []
    assert port_for("funnel", BASE) not in h.policies
    assert h.manager.running() == {}

    adapter.refusal = None
    await h.manager.start("funnel")
    assert "bind" in h.events, (
        "with the refusal withdrawn the same start must reach the bind -- "
        "otherwise the assertions above prove nothing about the refusal")


@pytest.mark.asyncio
async def test_a_bind_collision_is_reported_as_a_transport_error(h, monkeypatch):
    """Fix round 5, Important 3: `_bind` raises a raw `OSError` on a
    bind-collision race, and `errors.py` maps a raw `OSError` to 404 while
    this method's own docstring -- and `routes/transports.py`'s -- both
    promise `TransportError` (409) for every refusal. A caller must never
    see the raw `OSError` escape `start()`."""
    def _refuse(port, host="127.0.0.1"):
        raise OSError("[WinError 10048] address already in use")

    monkeypatch.setattr(server_mod, "bind_listener", _refuse)
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)

    with pytest.raises(TransportError) as excinfo:
        await h.manager.start("funnel")

    assert "18787" in str(excinfo.value) or "bind" in str(excinfo.value).lower()
    assert h.spawned == []
    assert port_for("funnel", BASE) not in h.policies
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_a_tunnel_that_never_announces_a_hostname_does_not_serve(h):
    """Spec §4: 'a registered socket with no published host is a door with no
    lock on the far side.'

    `TransportSession.hostname` is `str | None`, so this invariant lives in
    the manager rather than in the type (controller Ruling 13a) -- which is
    why it gets its own test asserting the whole teardown.
    """
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter, lines=[b"starting up\n", b"still starting\n"])

    with pytest.raises(TransportError) as excinfo:
        await h.manager.start("funnel")

    assert "hostname" in str(excinfo.value)
    assert h.spawned, "precondition: the tunnel really was spawned"
    assert "publish" not in h.events, "a nameless session reached publish()"
    assert h.served == [], "a nameless session was served"
    assert port_for("funnel", BASE) not in h.policies
    assert h.published.owners() == frozenset()
    assert h.sockets[0].fileno() == -1, "the socket was not closed"
    assert h.processes[0].returncode is not None, "the subprocess was not reaped"
    assert h.manager.running() == {}
    assert "funnel" not in h.listeners.tasks
    assert "funnel" not in h.listeners.sockets


@pytest.mark.asyncio
async def test_a_tunnel_that_stalls_past_the_timeout_does_not_serve(h,
                                                                   monkeypatch):
    """The other half of 'never announces': a process that says nothing at
    all and never closes its output. Bounded by `HOSTNAME_TIMEOUT_SECONDS`,
    shortened here so the test costs milliseconds instead of thirty seconds.
    """
    monkeypatch.setattr(manager_mod, "HOSTNAME_TIMEOUT_SECONDS", 0.05)
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter, lines=[])
    h.stall.add("funnel")

    with pytest.raises(TransportError):
        await h.manager.start("funnel")

    assert h.served == []
    assert port_for("funnel", BASE) not in h.policies
    assert h.sockets[0].fileno() == -1
    assert h.processes[0].returncode is not None


@pytest.mark.asyncio
async def test_a_chatty_process_that_never_announces_is_bounded(h, monkeypatch):
    """A hostile or merely verbose provider must not be a memory story.

    Asserting on `reads` rather than on the exception is the point: a manager
    that read every line and *then* gave up would raise the same error while
    holding the whole stream.
    """
    monkeypatch.setattr(manager_mod, "_MAX_SCAN_BYTES", 200)
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter, lines=[b"x" * 50 + b"\n" for _ in range(100)])

    with pytest.raises(TransportError):
        await h.manager.start("funnel")

    proc = h.processes[0]
    assert proc.stdout.reads < 100, (
        f"scanned {proc.stdout.reads} lines -- the byte bound did not hold")


@pytest.mark.asyncio
async def test_starting_a_transport_twice_is_refused(h):
    """And the refusal must not touch the session already running -- an
    'already running' path that ran the teardown would take the live tunnel
    down for the sake of the second caller's error message."""
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)

    session = await h.manager.start("funnel")

    with pytest.raises(TransportError, match="already running"):
        await h.manager.start("funnel")

    assert len(h.spawned) == 1
    assert sorted(h.manager.running()) == ["funnel"]
    assert h.manager.running()["funnel"].owner == session.owner
    assert session.sock.fileno() != -1, "the refusal closed the live socket"
    assert h.published.hosts_for(session.port) == frozenset(
        {"funnel.example.com"})


@pytest.mark.asyncio
async def test_starting_an_unknown_transport_is_refused(h):
    with pytest.raises(TransportError, match="nonesuch"):
        await h.manager.start("nonesuch")
    assert h.spawned == []
    assert h.sockets == []


# ═════════════════════════════════════════════════════════════════════════
# KI-17 layer 1 -- against the real argument vector
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["tailnet", "funnel"])
@pytest.mark.asyncio
async def test_the_spawned_argv_targets_a_registered_port_of_this_adapters_policy(
        h, monkeypatch, name):
    """Spec §2.4's third test, on the real command line of the real adapters.

    The target port is parsed out of the argv here, independently of the
    manager's own parser -- asserting the manager's check with the manager's
    own helper would only prove it agrees with itself.
    """
    _use_real_adapters(h, monkeypatch)

    session = await h.manager.start(name)

    argv = h.spawned[0]
    targets = [int(m.group(1)) for token in argv
               for m in [_LOOPBACK_TARGET_RE.search(token)] if m]
    assert targets == [port_for(name, BASE)], argv
    assert h.policies[targets[0]] == name
    assert targets[0] != local_port(BASE)
    assert session.port == targets[0]
    assert "shell" not in argv[0]


@pytest.mark.asyncio
async def test_the_spawned_argv_never_targets_the_local_port(h):
    """KI-17 itself: a tunnel pointed at the loopback listener's own port
    would inherit `POLICIES['local']` -- admin, bearer, `EXECUTE`.

    The honest start first is the vacuity guard: it proves this harness does
    spawn, so `len(h.spawned) == 1` afterwards means the liar was stopped and
    not that nothing ever spawns here.

    The refusal's *wording* is asserted, and that is not pedantry. The first
    version of this test only checked that the port number appeared in the
    message, and it passed with KI-17's own check deleted -- because the
    sibling 'registered to another listener' check refuses the same argv and
    names the same port. A test that cannot tell which check fired does not
    test either of them. The second half goes further: with local's own
    registry entry removed, the sibling check has nothing to say, so only
    KI-17's dedicated check can still refuse.
    """
    honest = _FakeAdapter("tailnet", events=h.events)
    h.register(honest)
    await h.manager.start("tailnet")
    assert len(h.spawned) == 1

    liar = _FakeAdapter("funnel", events=h.events,
                        target_port=local_port(BASE))
    h.register(liar)

    with pytest.raises(TransportError) as excinfo:
        await h.manager.start("funnel")

    assert str(local_port(BASE)) in str(excinfo.value)
    assert "loopback listener's own port" in str(excinfo.value), (
        f"some other check refused this argv: {excinfo.value}")
    assert len(h.spawned) == 1, "the liar's argv reached a subprocess"
    assert port_for("funnel", BASE) not in h.policies
    assert h.policies[local_port(BASE)] == "local", (
        "the local listener's own registry entry was disturbed")
    assert h.sockets[-1].fileno() == -1, "the funnel socket was not handed back"

    del h.policies[local_port(BASE)]
    with pytest.raises(TransportError) as unregistered:
        await h.manager.start("funnel")
    assert "loopback listener's own port" in str(unregistered.value), (
        "with local unregistered, only KI-17's own check can refuse this -- "
        f"and it did not: {unregistered.value}")
    assert len(h.spawned) == 1


@pytest.mark.asyncio
async def test_an_argv_targeting_another_transports_port_is_refused(h):
    """The rest of layer 1: a port that *is* registered, but to somebody else.
    A funnel forwarding into the tailnet listener would carry funnel's reach
    with tailnet's raisable ceiling."""
    tailnet = _FakeAdapter("tailnet", events=h.events)
    h.register(tailnet)
    await h.manager.start("tailnet")

    liar = _FakeAdapter("funnel", events=h.events,
                        target_port=port_for("tailnet", BASE))
    h.register(liar)

    with pytest.raises(TransportError) as excinfo:
        await h.manager.start("funnel")

    assert "tailnet" in str(excinfo.value)
    assert len(h.spawned) == 1
    assert port_for("funnel", BASE) not in h.policies


# ═════════════════════════════════════════════════════════════════════════
# The hostname, and the stream it arrives on
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_hostname_announced_only_on_stderr_is_read(h, monkeypatch):
    """`cloudflared` writes the line carrying its URL to **stderr** (Task 8).

    A manager watching stdout alone waits out the whole timeout, tears down,
    and kills a healthy tunnel -- and no fake can catch it, because a fake
    yields whatever the test feeds it. So this one runs a real subprocess:
    `sys.executable`, not a tunnel. The interpreter prints to stderr and then
    sleeps, so the announcement genuinely arrives on the other stream and the
    process is genuinely still alive when the manager publishes.
    """
    monkeypatch.setattr(manager_mod, "_spawn", h.real_spawn)

    script = ("import sys, time\n"
              "print('https://stderr.example.com', file=sys.stderr, flush=True)\n"
              "time.sleep(30)\n")

    class _StderrAdapter(_FakeAdapter):
        def command(self, port: int) -> list[str]:
            return [sys.executable, "-c", script, f"http://127.0.0.1:{int(port)}"]

    adapter = _StderrAdapter("funnel", events=h.events)
    h.register(adapter)

    session = await h.manager.start("funnel")
    try:
        assert session.hostname == "stderr.example.com"
        assert h.published.hosts_for(session.port) == frozenset(
            {"stderr.example.com"})
    finally:
        await h.manager.stop("funnel")

    assert session.process.returncode is not None, (
        "the real subprocess was not reaped by the teardown")


# ═════════════════════════════════════════════════════════════════════════
# Stop
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_stop_unpublishes_deregisters_drops_raises_and_closes_the_socket(h):
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)
    session = await h.manager.start("funnel")
    h.raises.grant("dev-1", "funnel", frozenset({Capability.OBSERVE}), 60,
                   "test", "why")

    # Preconditions, so the assertions below cannot pass against a manager
    # whose start never did anything in the first place.
    assert h.published.hosts_for(session.port) == frozenset(
        {"funnel.example.com"})
    assert h.policies[session.port] == "funnel"
    assert h.raises.capabilities_for("dev-1", "funnel") == frozenset(
        {Capability.OBSERVE})
    assert session.sock.fileno() != -1
    assert not session.serve_task.done()

    await h.manager.stop("funnel")

    assert h.published.hosts_for(session.port) == frozenset()
    assert h.published.owners() == frozenset()
    assert session.port not in h.policies
    assert h.raises.capabilities_for("dev-1", "funnel") == frozenset()
    assert session.serve_task.done()
    assert session.sock.fileno() == -1
    assert session.process.returncode is not None
    assert h.manager.running() == {}
    assert "funnel" not in h.listeners.tasks
    assert "funnel" not in h.listeners.sockets


@pytest.mark.asyncio
async def test_stop_is_safe_to_run_twice(h):
    """A crash handler and an orderly shutdown both call the teardown, and
    both may call it twice."""
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)
    session = await h.manager.start("funnel")

    await h.manager.stop("funnel")
    assert session.sock.fileno() == -1, (
        "precondition: the first stop must actually have torn down")

    await h.manager.stop("funnel")

    assert h.manager.running() == {}
    assert session.port not in h.policies
    assert h.published.owners() == frozenset()


@pytest.mark.asyncio
async def test_stopping_a_transport_that_never_started_is_not_an_error(h):
    await h.manager.stop("funnel")
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_a_crashed_tunnel_runs_the_same_teardown_as_an_orderly_stop(h):
    """Asserted as an equality between two runs of one transport in one
    harness -- the crash first, the orderly stop second -- so 'the same
    teardown' is a comparison rather than two lists of assertions that could
    drift apart.
    """
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)

    crashed = await h.manager.start("funnel")
    h.raises.grant("dev-1", "funnel", frozenset({Capability.OBSERVE}), 60,
                   "test", "why")
    assert h.raises.capabilities_for("dev-1", "funnel")  # precondition
    crashed.process.exit(1)
    # The session stops being reported as running the instant the teardown
    # takes it, so the wait is for the *last* observable step -- otherwise
    # the effects below would be read halfway through the unwind.
    assert await _settle(lambda: not h.manager.running()
                         and crashed.sock.fileno() == -1), (
        "the crashed tunnel's listener was left running")
    after_crash = _effects(h, "funnel", crashed.port, crashed.sock,
                           crashed.process, crashed.serve_task, "dev-1")

    h.lines_for["funnel"] = [b"https://second.example.com\n"]
    orderly = await h.manager.start("funnel")
    h.raises.grant("dev-1", "funnel", frozenset({Capability.OBSERVE}), 60,
                   "test", "why")
    await h.manager.stop("funnel")
    after_stop = _effects(h, "funnel", orderly.port, orderly.sock,
                          orderly.process, orderly.serve_task, "dev-1")

    assert after_crash == after_stop
    assert after_stop == {
        "published": frozenset(),
        "owners": frozenset(),
        "registered": False,
        "parked_task": False,
        "parked_socket": False,
        "raise": frozenset(),
        "socket_closed": True,
        "serve_task_done": True,
        "process_reaped": True,
        "running": [],
    }


@pytest.mark.asyncio
async def test_a_daemonised_spawn_process_exiting_does_not_stop_the_tunnel(h):
    """The other side of the crash path, and the reason it cannot simply
    watch every process: `tailscale serve --bg` daemonises, so the process
    TENKA spawned exits *on success*. A manager that read that as a crash
    would tear its own healthy tunnel down seconds after starting it."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)
    session = await h.manager.start("tailnet")

    session.process.exit(0)
    for _ in range(20):
        await asyncio.sleep(0.005)

    assert sorted(h.manager.running()) == ["tailnet"]
    assert h.policies[session.port] == "tailnet"
    assert h.published.hosts_for(session.port) == frozenset(
        {"tailnet.example.com"})
    assert session.sock.fileno() != -1


@pytest.mark.asyncio
async def test_two_sessions_of_one_transport_get_different_owners(h):
    """The hostname-reuse class `PublishedHosts` exists to prevent: two runs
    of one tunnel get different names from the provider, so the second must
    not inherit the first's ownership -- or stopping the second leaves the
    first's name trusted forever."""
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)

    first = await h.manager.start("funnel")
    await h.manager.stop("funnel")

    h.lines_for["funnel"] = [b"https://second.example.com\n"]
    second = await h.manager.start("funnel")

    assert first.owner != second.owner
    assert first.owner != "funnel" and second.owner != "funnel"
    assert h.published.hosts_for(second.port) == frozenset(
        {"second.example.com"}), "the first run's hostname is still trusted"

    await h.manager.stop("funnel")
    assert h.published.hosts_for(second.port) == frozenset()


@pytest.mark.asyncio
async def test_stop_all_leaves_only_the_local_listener(h):
    for name in ("tailnet", "funnel"):
        h.register(_FakeAdapter(name, events=h.events))
        await h.manager.start(name)

    assert sorted(h.manager.running()) == ["funnel", "tailnet"]
    assert len(h.policies) == 3

    failures = await h.manager.stop_all()

    assert failures == []
    assert h.manager.running() == {}
    assert dict(h.policies) == {BASE: "local"}
    assert list(h.listeners.tasks) == ["local"]
    assert list(h.listeners.sockets) == ["local"]
    assert all(sock.fileno() == -1 for sock in h.sockets)
    assert all(proc.returncode is not None for proc in h.processes)
    assert h.published.owners() == frozenset()


# ═════════════════════════════════════════════════════════════════════════
# The stop that must verify it stopped
# ═════════════════════════════════════════════════════════════════════════

_STOP_ARGV = ["fake-tailnet", "serve", "--https", "9443", "off"]


@pytest.mark.asyncio
async def test_a_daemonising_transports_stop_runs_its_argv_and_verifies_it(h):
    """`tailscale serve --bg` daemonises: the process TENKA spawned exits on
    its own, so terminating it again touches nothing and only the argv
    `stop_command()` returns un-serves the mapping."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)
    await h.manager.start("tailnet")

    await h.manager.stop("tailnet")

    assert _STOP_ARGV in h.ran, "the provider's own stop command never ran"
    assert ["fake-tailnet", "serve", "status", "--json"] in h.ran, (
        "the stop was never verified against the provider's own status")


@pytest.mark.asyncio
async def test_a_stop_that_leaves_a_mapping_under_our_public_port_fails_loudly(h):
    """The worst outcome this milestone can produce is an internet-facing
    listener that silently stayed up. `tailscale`'s `off` form is documented
    but unexercised, and `<target>` accepts free text -- so a mis-parsed stop
    can exit 0 with the mapping still live."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)
    session = await h.manager.start("tailnet")
    h.status_payload = {
        "Web": {"box.tail1234.ts.net:9443": {
            "Handlers": {"/": {"Proxy": f"http://127.0.0.1:{session.port}"}}}}}

    with pytest.raises(TransportError) as excinfo:
        await h.manager.stop("tailnet")

    assert "9443" in str(excinfo.value)
    # Loud, but not a leak: everything else was still torn down.
    assert session.port not in h.policies
    assert h.published.owners() == frozenset()
    assert session.sock.fileno() == -1
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_a_stop_whose_status_cannot_be_read_is_reported_unverified(h,
                                                                        monkeypatch):
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)
    await h.manager.start("tailnet")
    h.status_payload = "not a document"

    with pytest.raises(TransportError, match="UNVERIFIED"):
        await h.manager.stop("tailnet")


@pytest.mark.asyncio
async def test_a_foreground_transports_stop_runs_no_second_command(h):
    """`stop_command()` returning `None` is a real case, not an oversight:
    `cloudflared tunnel --url` runs in the foreground, so reaping the process
    *is* the stop."""
    adapter = _FakeAdapter("funnel", events=h.events, stop_argv=None)
    h.register(adapter)
    session = await h.manager.start("funnel")

    await h.manager.stop("funnel")

    assert h.ran == [], f"a foreground transport ran {h.ran}"
    assert session.process.returncode is not None
    assert session.process.terminated == 1


@pytest.mark.asyncio
async def test_a_process_that_ignores_kill_makes_stop_report_failure(h, monkeypatch):
    """Critical 1, fix round 5: for a foreground provider (`stop_command()`
    returns `None`), reaping the process *is* the whole stop -- so if it
    ignores both `terminate()` and the escalation to `kill()`, `_reap` must
    say so rather than returning normally. A prior version returned
    normally here, so `stop()` reported success while the tunnel process --
    and, for a real provider, its live edge connection -- kept running.
    Bounded by `wait_for` so a regression fails rather than hangs."""
    monkeypatch.setattr(manager_mod, "_REAP_TIMEOUT_SECONDS", 0.05)
    adapter = _FakeAdapter("funnel", events=h.events, stop_argv=None)
    h.register(adapter)
    session = await h.manager.start("funnel")

    # Make the spawned process immune to both `terminate()` and `kill()`.
    monkeypatch.setattr(session.process, "terminate", lambda: None)
    monkeypatch.setattr(session.process, "kill", lambda: None)

    with pytest.raises(TransportError, match="UNVERIFIED"):
        await asyncio.wait_for(h.manager.stop("funnel"), 5.0)

    assert session.process.returncode is None, (
        "the stuck process was falsely reported as reaped")


@pytest.mark.asyncio
async def test_stop_all_reports_a_failed_verification_without_stranding_others(h):
    """A shutdown must not be blocked by one transport that will not stop --
    but the failure must still be reported, not swallowed."""
    h.register(_FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV))
    h.register(_FakeAdapter("funnel", events=h.events))
    tailnet = await h.manager.start("tailnet")
    funnel = await h.manager.start("funnel")
    h.status_payload = {"Web": {"box.ts.net:9443": {}}}

    failures = await h.manager.stop_all()

    assert len(failures) == 1 and "9443" in failures[0]
    assert h.manager.running() == {}
    assert funnel.sock.fileno() == -1, "the healthy transport was stranded"
    assert tailnet.sock.fileno() == -1
    assert dict(h.policies) == {BASE: "local"}


# ═════════════════════════════════════════════════════════════════════════
# The invariant that moved out of the type system (controller Ruling 13a)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_session_with_no_hostname_is_never_treated_as_serving(h):
    """`TransportSession.hostname` is `str | None`, filled in after the
    announcement via `dataclasses.replace`, so spec §4's 'a tunnel that never
    announces a hostname does not serve' is enforced procedurally rather than
    by the type. An invariant that moved out of the type system needs its own
    test."""
    port = port_for("funnel", BASE)
    sock = server_mod.bind_listener(port)
    h.sockets.append(sock)
    task = _sleeping_task("funnel")
    h.serve_tasks.append(task)
    nameless = TransportSession(policy_name="funnel", port=port, owner="o",
                                process=_FakeProcess([]), sock=sock,
                                serve_task=task, hostname=None)
    named = replace(nameless, hostname="funnel.example.com")

    assert is_serving(nameless) is False
    assert is_serving(named) is True, (
        "precondition: a session that *did* announce must be serving, or the "
        "assertion above holds for the wrong reason")

    with pytest.raises(TransportError):
        require_serving(nameless)
    assert require_serving(named) is named

    h.manager._sessions["funnel"] = nameless
    assert h.manager.running() == {}, (
        "a nameless session was reported as a running transport")


# ═════════════════════════════════════════════════════════════════════════
# Fix round 1
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_stop_arriving_while_a_transport_is_still_starting_is_not_lost(h):
    """Fix round 1, Important 1. The window is `HOSTNAME_TIMEOUT_SECONDS`
    wide: a start between its spawn and its publish is in `_starting` and
    nowhere else, so a stop that only consults `_sessions` returns having done
    nothing, the caller is told it succeeded, and the tunnel comes up
    afterwards. A daemonising provider is used deliberately -- the assertion
    that its `off` argv ran is the one that says the mapping is gone, not
    merely that TENKA stopped watching it.
    """
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[])
    h.stall.add("tailnet")

    starting = asyncio.create_task(h.manager.start("tailnet"))
    assert await _settle(lambda: bool(h.spawned)), "the start never spawned"
    assert "tailnet" not in h.manager.running(), (
        "precondition: the session is not tracked yet -- that is the window")

    await h.manager.stop("tailnet")

    # Bounded, so a manager that cannot see the starting transport fails here
    # instead of hanging this test forever.
    with pytest.raises(TransportError, match="still starting"):
        await asyncio.wait_for(starting, 2.0)

    assert _STOP_ARGV in h.ran, (
        "the provider's own mapping was never un-served -- this is the "
        "shutdown case that leaves a tunnel published after TENKA has exited")
    assert h.manager.running() == {}
    assert port_for("tailnet", BASE) not in h.policies
    assert h.published.owners() == frozenset()
    assert h.sockets[0].fileno() == -1
    assert h.processes[0].returncode is not None
    assert "tailnet" not in h.listeners.tasks


@pytest.mark.asyncio
async def test_stop_all_stops_a_transport_that_is_still_starting(h):
    """The version that matters: an operator quitting TENKA inside that
    window."""
    adapter = _FakeAdapter("funnel", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[])
    h.stall.add("funnel")

    starting = asyncio.create_task(h.manager.start("funnel"))
    assert await _settle(lambda: bool(h.spawned))

    failures = await h.manager.stop_all()

    with pytest.raises(TransportError):
        await asyncio.wait_for(starting, 2.0)
    assert failures == [], failures
    assert _STOP_ARGV in h.ran
    assert dict(h.policies) == {BASE: "local"}
    assert h.sockets[0].fileno() == -1
    assert h.processes[0].returncode is not None


@pytest.mark.asyncio
async def test_a_cancelled_start_does_not_orphan_its_subprocess(h, monkeypatch):
    """A stop landing after the child exists but before its handle comes back
    must not leave a process nobody holds -- an orphan tunnel is exactly the
    outcome a cancellable start exists to prevent. So the spawn is shielded
    and re-awaited on cancellation.

    The window is made deterministic rather than raced for: the spawn creates
    the real child, announces itself, and only then sleeps. A cancel arriving
    during that sleep is the case. With the shield removed, the handle is
    never returned, the teardown reaps nothing, and the assertion on the
    child's return code fails -- which is what makes this a test of the shield
    and not of the timing.
    """
    created: list = []
    spawned_child = asyncio.Event()
    script = "import time; time.sleep(5)"

    async def _slow_spawn(argv):
        proc = await h.real_spawn(argv)
        created.append(proc)
        spawned_child.set()
        await asyncio.sleep(0.05)
        return proc

    monkeypatch.setattr(manager_mod, "_spawn", _slow_spawn)

    class _SlowSpawnAdapter(_FakeAdapter):
        def command(self, port: int) -> list[str]:
            return [sys.executable, "-c", script,
                    f"http://127.0.0.1:{int(port)}"]

    h.register(_SlowSpawnAdapter("funnel", events=h.events), lines=[])

    starting = asyncio.create_task(h.manager.start("funnel"))
    await asyncio.wait_for(spawned_child.wait(), 5.0)

    await h.manager.stop("funnel")
    with pytest.raises(TransportError):
        await asyncio.wait_for(starting, 5.0)

    assert created, "precondition: a real child was spawned"
    assert created[0].returncode is not None, (
        "the spawned child was orphaned -- the start was cancelled before its "
        "handle came back, so the teardown had nothing to reap")
    assert h.manager.running() == {}
    assert port_for("funnel", BASE) not in h.policies


@pytest.mark.parametrize("payload", [
    {"Web": "not an object"},
    {"Web": {}, "Foreground": 5},
    {"Foreground": {"session": {"Web": 7}}},
    {"Foreground": {"session": "not an object"}},
])
@pytest.mark.asyncio
async def test_a_status_document_of_an_unrecognised_shape_is_unverified(h,
                                                                       payload):
    """Fix round 1, Important 2. Task 7's preflight may degrade an
    unrecognised document to a warning because KI-17 layer 3 is behind it.
    This check has nothing behind it, so a document it cannot sweep is an
    UNVERIFIED stop, never a verified one."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)
    session = await h.manager.start("tailnet")
    h.status_payload = payload

    with pytest.raises(TransportError, match="UNVERIFIED"):
        await h.manager.stop("tailnet")

    # Loud, not a leak: the rest of the teardown still ran.
    assert session.port not in h.policies
    assert session.sock.fileno() == -1
    assert h.published.owners() == frozenset()


@pytest.mark.asyncio
async def test_an_empty_status_document_is_a_verified_stop(h):
    """The other side of the test above, and the ordinary case: this machine's
    real cleared `tailscale serve status --json` prints a bare `{}`. If that
    were UNVERIFIED, every successful stop would report a failure."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)
    await h.manager.start("tailnet")
    h.status_payload = {}

    await h.manager.stop("tailnet")  # must not raise

    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_the_status_reader_is_the_adapters_own(h):
    """Fix round 1, Minor 3. The reader is named by the adapter, not derived
    from the stop argv's verb -- deriving it asked a funnel-scoped view a
    serve-scoped question, and assumed every provider's CLI has the shape
    `<binary> <verb> status --json`."""
    reader = ["fake-tailnet", "whatever-this-provider-calls-it", "--json"]
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV,
                           status_argv=reader)
    h.register(adapter)
    await h.manager.start("tailnet")

    await h.manager.stop("tailnet")

    assert reader in h.ran, h.ran
    assert ["fake-tailnet", "serve", "status", "--json"] not in h.ran, (
        "the reader was derived from the stop argv rather than asked for")


@pytest.mark.asyncio
async def test_a_stop_with_no_status_command_is_reported_unverified(h):
    """An adapter that names a stop but no way to check it cannot have its
    stop believed -- the pair is what makes a stop verifiable."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV,
                           status_argv=None)
    h.register(adapter)
    await h.manager.start("tailnet")

    with pytest.raises(TransportError, match="UNVERIFIED"):
        await h.manager.stop("tailnet")


@pytest.mark.asyncio
async def test_a_start_refused_before_its_spawn_runs_no_provider_stop(h):
    """Fix round 1, Minor 4. A layer-1 refusal happens before the subprocess
    exists, so there is no mapping of TENKA's own to remove -- and running the
    `off` argv anyway would destroy whatever the operator already had under
    that public port, which spec §2.3 L2 forbids ('does not silently
    repair')."""
    liar = _FakeAdapter("funnel", events=h.events, stop_argv=_STOP_ARGV,
                        target_port=local_port(BASE))
    h.register(liar)

    with pytest.raises(TransportError):
        await h.manager.start("funnel")

    assert h.ran == [], f"a refused start ran {h.ran} against the provider"
    assert h.spawned == []

    # Vacuity guard: the same shape of adapter, telling the truth, does reach
    # the provider's stop -- so `h.ran == []` above is the refusal, not the
    # harness never running anything.
    honest = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(honest)
    await h.manager.start("tailnet")
    await h.manager.stop("tailnet")
    assert _STOP_ARGV in h.ran


@pytest.mark.asyncio
async def test_an_unwind_that_cannot_prove_the_mapping_gone_says_so(h):
    """Fix round 1, Minor 5. A 409 reading only 'never announced a hostname'
    hides the half the operator has to act on: that the provider-side mapping
    could not be proved gone."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[b"nothing useful\n"])
    h.status_payload = {"Web": {"box.ts.net:9443": {}}}

    with pytest.raises(TransportError) as excinfo:
        await h.manager.start("tailnet")

    message = str(excinfo.value)
    assert "never announced a hostname" in message
    assert "STILL keyed" in message and "9443" in message, message


@pytest.mark.asyncio
async def test_a_raise_that_cannot_be_dropped_is_reported(h):
    """Fix round 1, Minor 6. A swallowed `drop_policy` failure leaves a raise
    scoped to a policy name the next session reuses -- the exact inheritance
    `drop_policy` exists to prevent."""
    class _AngryRaises:
        def drop_policy(self, policy_name: str) -> None:
            raise RuntimeError("the raise store is wedged")

    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter)
    session = await h.manager.start("funnel")
    h.app.state.raises = _AngryRaises()

    with pytest.raises(TransportError, match="raises could not be dropped"):
        await h.manager.stop("funnel")

    assert session.sock.fileno() == -1, "the rest of the teardown was abandoned"
    assert session.port not in h.policies
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_a_stop_uses_the_adapter_its_start_used(h):
    """Fix round 1, Minor 8. Re-resolving the adapter by name at stop time
    means a registry mutated in between runs a different provider's stop argv
    -- against a mapping that provider never made."""
    original = _FakeAdapter("funnel", events=h.events, stop_argv=_STOP_ARGV)
    h.register(original)
    await h.manager.start("funnel")

    impostor = _FakeAdapter("funnel", events=h.events,
                            stop_argv=["fake-impostor", "serve", "off"])
    h.registry.reset()
    h.registry.register("funnel", impostor)

    await h.manager.stop("funnel")

    assert _STOP_ARGV in h.ran
    assert impostor.stop_calls == [], "the impostor's stop command was run"


@pytest.mark.asyncio
async def test_a_stop_landing_inside_a_teardown_does_not_leave_it_half_done(
        h, monkeypatch):
    """Found while testing Important 1's fix, and in the same family as it.

    A stop cancels an in-flight start; if that start is already unwinding, the
    cancellation lands *inside* the teardown -- and a teardown abandoned
    between the socket close and the provider's `off` argv leaves exactly what
    it exists to prevent, while the stop that caused it reports success. So
    the teardown runs in a task of its own and is re-awaited.

    The window is made deterministic: the provider's stop command blocks in a
    worker thread until this test has fired the cancel.

    The exception `starting` raises changed in fix round 4:
    `_start_and_unwind`'s except block now honours its own task's
    outstanding cancellation (from the `stop()` below) once its teardown is
    actually done, in place of re-raising the natural "never announced a
    hostname" failure that was already in flight when the stop arrived --
    matching the other two call sites' priority (a cancellation, once its
    protected work has completed, outranks the failure that happened to be
    in progress at the time). The failure is not lost: `_unwind_failures`
    still carries it, and every assertion below on the teardown's own
    completeness is unchanged and still the point of this test.
    """
    ran_stop = threading.Event()
    real_run = manager_mod._run_argv

    def _slow_run(argv):
        h.ran.append(list(argv))
        ran_stop.set()
        time.sleep(0.15)
        if "--json" in argv:
            return 0, json.dumps(h.status_payload)
        return 0, ""

    monkeypatch.setattr(manager_mod, "_run_argv", _slow_run)
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[b"nothing useful\n"])

    starting = asyncio.create_task(h.manager.start("tailnet"))
    assert await _settle(lambda: ran_stop.is_set()), (
        "precondition: the unwind must already be inside its teardown")

    await h.manager.stop("tailnet")

    with pytest.raises(TransportError, match="still starting"):
        await asyncio.wait_for(starting, 5.0)

    assert _STOP_ARGV in h.ran
    assert ["fake-tailnet", "serve", "status", "--json"] in h.ran, (
        "the teardown was abandoned before it could verify the stop")
    assert h.processes[0].returncode is not None, (
        "the teardown was abandoned before it reaped the subprocess")
    assert h.sockets[0].fileno() == -1
    assert port_for("tailnet", BASE) not in h.policies
    assert h.manager.running() == {}


# ═════════════════════════════════════════════════════════════════════════
# Fix round 2
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_two_concurrent_stops_of_a_starting_transport_still_tear_it_down(
        h, monkeypatch):
    """Fix round 2, Important. Two stops inside the start window -- two clicks
    on a route, or a `DELETE` racing `stop_all`.

    A second `cancel()` on the start task is the hazard: delivered at the
    wrong moment it lands on the teardown's own re-await, where
    `CancelledError` passes straight through `suppress(Exception)`, and the
    teardown is abandoned after its first three steps -- no `off` argv, no
    verification, no reap -- while both stops report success, because the
    abandoned unwind never recorded its failures. That is Important 1's own
    outcome, reached through the mechanism that fixed it.

    **The assertion is on the count of cancellation requests, not only on the
    outcome, and that is deliberate.** Measured on this interpreter, the second
    cancel is absorbed by `asyncio.shield` -- it cancels the shield's outer
    future, the start task moves to its recovery re-await, and the teardown
    survives; it takes a *third* stop to reach the re-await itself. So an
    outcome-only test passes with the guard deleted, which is a test proving
    nothing (it did, on the first attempt at this one). What the guard promises
    is exactly that a second stop requests no second cancellation, and
    `Task.cancelling()` is that count -- interpreter-independent, and one
    assertion instead of a three-deep interleaving.

    The window is deterministic rather than raced for: the provider's stop
    command blocks in a worker thread until this test has issued the second
    stop.
    """
    ran_stop = threading.Event()

    def _slow_run(argv):
        h.ran.append(list(argv))
        ran_stop.set()
        time.sleep(0.2)
        if "--json" in argv:
            return 0, json.dumps(h.status_payload)
        return 0, ""

    monkeypatch.setattr(manager_mod, "_run_argv", _slow_run)
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[])
    h.stall.add("tailnet")

    starting = asyncio.create_task(h.manager.start("tailnet"))
    assert await _settle(lambda: bool(h.spawned)), "the start never spawned"

    start_task = h.manager._starting["tailnet"]
    first = asyncio.create_task(h.manager.stop("tailnet"))
    assert await _settle(lambda: ran_stop.is_set()), (
        "precondition: the first stop's unwind must be inside the teardown")
    assert start_task.cancelling() == 1, (
        "precondition: the first stop must have cancelled the start exactly "
        f"once, not {start_task.cancelling()} times")

    await asyncio.wait_for(h.manager.stop("tailnet"), 5.0)

    assert start_task.cancelling() == 1, (
        "the second stop requested another cancellation -- delivered at the "
        "wrong moment that abandons the teardown between the socket close and "
        "the provider's own stop, and both stops report success")
    await asyncio.wait_for(first, 5.0)
    with pytest.raises(TransportError):
        await asyncio.wait_for(starting, 5.0)

    assert _STOP_ARGV in h.ran
    assert ["fake-tailnet", "serve", "status", "--json"] in h.ran, (
        "the second stop aborted the teardown before it verified the stop -- "
        "the provider mapping may still be up and nothing said so")
    assert h.processes[0].returncode is not None, (
        "the teardown was abandoned before it reaped the subprocess")
    assert h.sockets[0].fileno() == -1
    assert dict(h.policies) == {BASE: "local"}
    assert h.published.owners() == frozenset()
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_a_spawn_the_os_refused_runs_no_provider_stop(h, monkeypatch):
    """Fix round 2, Minor 1; the wrapping is fix round 3's fold-in. A spawn
    that raises created no mapping, so the `off` argv must not run against
    whatever the operator already had under that public port -- the gate
    follows whether a child may exist, not whether one was intended. The OS's
    own refusal (no such binary, exec denied) must not escape `start()` in
    its native exception type either: `start()`'s docstring promises
    `TransportError` on every refusal, and `cloudflared` being absent from
    this very machine makes a raw `FileNotFoundError` the commonest way that
    promise could be broken."""
    async def _refused(argv):
        h.spawned.append(list(argv))
        raise FileNotFoundError("the provider binary is not installed")

    monkeypatch.setattr(manager_mod, "_spawn", _refused)
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)

    with pytest.raises(TransportError, match="tailnet") as excinfo:
        await h.manager.start("tailnet")
    assert isinstance(excinfo.value.__cause__, FileNotFoundError), (
        "the OS's own refusal should be chained, not discarded, so an "
        "operator (or a test) can still see what actually failed")

    assert h.spawned, "precondition: the spawn really was attempted"
    assert h.ran == [], f"a refused spawn ran {h.ran} against the provider"
    assert port_for("tailnet", BASE) not in h.policies
    assert h.sockets[0].fileno() == -1
    assert h.manager.running() == {}

    # Vacuity guard: a spawn that succeeds does reach the provider's stop, so
    # `h.ran == []` above is the gate and not the harness.
    monkeypatch.setattr(manager_mod, "_spawn", h.fake_spawn)
    h.register(_FakeAdapter("funnel", events=h.events, stop_argv=_STOP_ARGV))
    await h.manager.start("funnel")
    await h.manager.stop("funnel")
    assert _STOP_ARGV in h.ran


@pytest.mark.asyncio
async def test_a_stop_in_the_first_tick_of_a_start_reports_no_stale_failure(h):
    """Fix round 2, Minor 2. `_unwind_failures` was cleared on the start
    task's first line, which is one tick after the task became visible to
    `stop`. A stop landing in that tick cancelled a task that never ran and
    reported the *previous* attempt's failures as this stop's."""
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[b"nothing useful\n"])
    h.status_payload = {"Web": {"box.ts.net:9443": {}}}

    with pytest.raises(TransportError) as first:
        await h.manager.start("tailnet")
    assert "STILL keyed" in str(first.value), (
        "precondition: the first attempt must have recorded a failure")

    h.status_payload = {}
    starting = asyncio.create_task(h.manager.start("tailnet"))
    # One turn: `start()` reaches `await task` and the start task is scheduled
    # but has not run a line of its own.
    await asyncio.sleep(0)

    # Must not raise: there is no failure belonging to *this* attempt.
    await asyncio.wait_for(h.manager.stop("tailnet"), 5.0)

    with pytest.raises(TransportError, match="still starting"):
        await asyncio.wait_for(starting, 5.0)


@pytest.mark.asyncio
async def test_a_cancelled_caller_is_answered_with_its_own_cancellation(h):
    """Fix round 2, Minor 3. The marker is not evidence about *whose*
    cancellation arrived: a stop may have set it while this caller's own task
    was independently cancelled, and answering that caller with a
    `TransportError` instead of a `CancelledError` is the shutdown-hang shape
    the code's own comment warns about.

    The marker is set directly here, which is what a stop would have done a
    tick earlier -- the point of the finding is that its presence proves
    nothing about the cancellation this frame is handling.
    """
    adapter = _FakeAdapter("funnel", events=h.events)
    h.register(adapter, lines=[])
    h.stall.add("funnel")

    caller = asyncio.create_task(h.manager.start("funnel"))
    assert await _settle(lambda: bool(h.spawned))
    h.manager._stopped_while_starting.add("funnel")

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, 5.0)


# ═════════════════════════════════════════════════════════════════════════
# Fix round 3
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [2, 3, 5, 9])
async def test_a_teardown_survives_any_number_of_repeat_cancellations(
        h, monkeypatch, depth):
    """Important 2. Round 1 and round 2 both shielded the teardown once and
    recovered with a single plain re-await -- which survives exactly one
    repeat cancellation of whatever is waiting on it before a further one
    reaches the teardown task directly and abandons it mid-flight (between
    the socket close and the provider's own stop, in this test's window).

    `depth` cancellations are fired at the task running `stop()` once it is
    already inside the teardown's protected wait, all before the provider's
    blocked stop command is allowed to return. At `depth == 1` even the
    single-recovery shape survives (that is what the shield alone already
    does); this starts at 2, which is where a reverted, single-recovery
    `_teardown_completely` first abandons the teardown -- confirmed by
    running this test against that reversion, not merely reasoned about.

    The window is gated on an `asyncio.Event`, not on a sleep: the fake stop
    command blocks until this test explicitly releases it, so every one of
    `depth` cancellations is provably delivered before the teardown is
    allowed to finish, however many there are.
    """
    loop = asyncio.get_running_loop()
    ran_stop = asyncio.Event()
    release = asyncio.Event()

    def _slow_run(argv):
        # Runs in `to_thread`'s worker thread, which has no running loop of
        # its own -- the loop captured above, not `get_running_loop()` here,
        # is what a thread-safe call back into it needs.
        h.ran.append(list(argv))
        if argv == _STOP_ARGV:
            loop.call_soon_threadsafe(ran_stop.set)
            # A plain `Event().wait()` (threading's, not asyncio's) belongs
            # here, but reaching for asyncio's own `Event` from a thread via
            # `run_coroutine_threadsafe` would dead-lock this same loop, so a
            # small poll loop stands in for a cross-thread wait instead.
            # Capped, not unbounded: a failed assertion below must not leave
            # this worker thread blocked forever (mutation testing routinely
            # fails the assertion right after this without ever releasing).
            deadline = time.monotonic() + 5.0
            while not release.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
        if "--json" in argv:
            return 0, json.dumps(h.status_payload)
        return 0, ""

    monkeypatch.setattr(manager_mod, "_run_argv", _slow_run)
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)
    await h.manager.start("tailnet")

    stop_task = asyncio.create_task(h.manager.stop("tailnet"))
    assert await _settle(lambda: ran_stop.is_set()), (
        "precondition: the teardown must already be running the off argv")

    for _ in range(depth):
        stop_task.cancel()
        await asyncio.sleep(0)

    assert not stop_task.done(), (
        f"the teardown was abandoned by the {depth}th cancellation instead "
        f"of running to completion -- a mapping may still be on the open "
        f"internet with nothing reporting it")

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stop_task, 5.0)

    assert _STOP_ARGV in h.ran
    assert ["fake-tailnet", "serve", "status", "--json"] in h.ran, (
        "the teardown was abandoned before it could verify the stop")
    assert h.processes[0].returncode is not None, (
        "the teardown was abandoned before it reaped the subprocess")
    assert h.sockets[0].fileno() == -1
    assert dict(h.policies) == {BASE: "local"}
    assert h.published.owners() == frozenset()
    assert h.manager.running() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [2, 3, 5, 9])
async def test_the_spawn_survives_any_number_of_repeat_cancellations_too(
        h, monkeypatch, depth):
    """Important 2, the other call site. The same shield-then-plain-re-await
    shape guarded `create_subprocess_exec` -- a cancel landing there once the
    OS has already created the child, but before the handle is assigned,
    orphans a real process this manager can never again reach. `depth`
    cancellations land on the task running `_start_and_unwind` while the
    spawn is still in flight; the process must still be recovered and reaped
    regardless of how many there are.
    """
    spawn_started = asyncio.Event()
    release_spawn = asyncio.Event()

    async def _slow_spawn(argv):
        h.spawned.append(list(argv))
        spawn_started.set()
        await release_spawn.wait()
        proc = _FakeProcess(h.lines_for.get("tailnet", []))
        h.processes.append(proc)
        return proc

    monkeypatch.setattr(manager_mod, "_spawn", _slow_spawn)
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter)

    outer_task = asyncio.create_task(h.manager.start("tailnet"))
    assert await _settle(lambda: spawn_started.is_set()), (
        "precondition: the spawn never started")
    inner_task = h.manager._starting["tailnet"]

    for _ in range(depth):
        inner_task.cancel()
        await asyncio.sleep(0)

    assert not inner_task.done(), (
        f"the spawn was abandoned by the {depth}th cancellation -- an orphan "
        f"process this manager can no longer reach may now exist")

    release_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(outer_task, 5.0)

    assert h.processes, "the process must have been recovered, not orphaned"
    assert h.processes[0].terminated or h.processes[0].killed, (
        "the recovered process was not reaped")
    assert port_for("tailnet", BASE) not in h.policies
    assert h.sockets[0].fileno() == -1
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_a_cancelled_stop_never_forwards_a_further_cancel_into_a_starting_transport(
        h):
    """Important 1. `asyncio.gather` cancels its child when the *awaiter* is
    cancelled, so a stop that is itself cancelled while it waits for an
    in-flight start to unwind would deliver a second cancellation into that
    start through the one call the round 2 guard does not cover.
    `asyncio.wait` never forwards, no matter how many times the wait itself
    is cancelled here.
    """
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[])
    h.stall.add("tailnet")

    starting = asyncio.create_task(h.manager.start("tailnet"))
    assert await _settle(lambda: bool(h.spawned)), "the start never spawned"
    start_task = h.manager._starting["tailnet"]

    stop_task = asyncio.create_task(h.manager.stop("tailnet"))
    assert await _settle(lambda: start_task.cancelling() >= 1)
    assert start_task.cancelling() == 1

    for _ in range(3):
        stop_task.cancel()
        await asyncio.sleep(0)

    assert start_task.cancelling() == 1, (
        "the stop's own cancellation reached the starting transport -- "
        "exactly what `asyncio.wait` (not `gather`) exists to stop")

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stop_task, 5.0)

    # The starting transport's own unwind must still complete regardless of
    # the cancelled stop above.
    with pytest.raises(TransportError, match="still starting"):
        await asyncio.wait_for(starting, 5.0)
    assert port_for("tailnet", BASE) not in h.policies
    assert h.sockets[0].fileno() == -1
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_stop_all_keeps_going_past_a_transport_whose_stop_was_cancelled(
        h, monkeypatch):
    """Important 1, secondary. `stop`'s own teardown only ever lets a
    cancellation reach its caller after that teardown has completed (see the
    depth tests above) -- so a `CancelledError` out of `stop` here means the
    transport in question is already down, only unreported. Before this
    round, `stop_all`'s per-name guard did not catch `CancelledError`, so it
    escaped the loop entirely and every transport after the cancelled one was
    never even attempted -- exactly the stranding `stop_all`'s own docstring
    says it must not do.
    """
    calls: list[str] = []

    async def _fake_stop(name):
        calls.append(name)
        if name == "tailnet":
            raise asyncio.CancelledError()

    monkeypatch.setattr(h.manager, "stop", _fake_stop)
    h.manager._sessions["tailnet"] = None
    h.manager._sessions["funnel"] = None

    failures = await h.manager.stop_all()

    assert calls == ["tailnet", "funnel"], (
        "stop_all abandoned the remaining transports after one stop was "
        f"cancelled -- only reached {calls}")
    assert any("tailnet" in f and "cancelled" in f for f in failures), (
        "the cancellation must still be reported, not silently dropped")


# ═════════════════════════════════════════════════════════════════════════
# Fix round 4
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2, 3, 5])
async def test_a_cancelled_stop_of_a_starting_transport_waits_for_its_teardown(
        h, monkeypatch, depth):
    """Important. The "still starting" branch cancels the in-flight start
    and then waited via a plain `asyncio.wait` -- which never forwards a
    cancel onto `starting` itself, however many times *that* wait is
    cancelled (Important 1's own mechanism, still correct), but does not
    loop, so the *first* cancellation of `stop`'s own caller (`stop_all`'s
    iterating task, or a route handler via anyio's per-tick redelivery --
    the same mechanism round 3 targeted for the sibling "already running"
    branch) propagates out of `stop` immediately, before the starting
    transport's own (correctly protected) teardown has necessarily
    finished.

    `stop_all`'s `except asyncio.CancelledError` handler then records that
    teardown as "already completed" -- true for the sibling branch (fix
    round 3), false for this one, since nothing re-awaited the abandoned
    task afterward. At shutdown, that gap is exactly the named worst
    outcome: a tunnel possibly still mapped to the internet, with nothing
    left running long enough to find out.

    `depth` cancellations are fired at the task calling `stop()`, all
    delivered once already inside the wait, while the starting transport's
    own teardown is deliberately held open on a gate.
    """
    loop = asyncio.get_running_loop()
    ran_stop = asyncio.Event()
    release = asyncio.Event()

    def _slow_run(argv):
        h.ran.append(list(argv))
        if argv == _STOP_ARGV:
            loop.call_soon_threadsafe(ran_stop.set)
            deadline = time.monotonic() + 5.0
            while not release.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
        if "--json" in argv:
            return 0, json.dumps(h.status_payload)
        return 0, ""

    monkeypatch.setattr(manager_mod, "_run_argv", _slow_run)
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[])
    h.stall.add("tailnet")

    starting = asyncio.create_task(h.manager.start("tailnet"))
    assert await _settle(lambda: bool(h.spawned)), "the start never spawned"
    start_task = h.manager._starting["tailnet"]

    stop_task = asyncio.create_task(h.manager.stop("tailnet"))
    assert await _settle(lambda: ran_stop.is_set()), (
        "precondition: the starting transport's own teardown must already "
        "be running the off argv")
    assert not start_task.done(), (
        "precondition: the starting transport's teardown is still in flight")

    for _ in range(depth):
        stop_task.cancel()
        await asyncio.sleep(0)

    assert not stop_task.done(), (
        f"stop() returned/raised after {depth} cancellation(s) of its own "
        f"caller while the starting transport's teardown was still "
        f"running -- stop_all would then report a stop as 'already "
        f"completed' when it was not")
    assert not start_task.done(), (
        "the starting transport's own teardown must still be running too")

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stop_task, 5.0)

    assert start_task.done(), (
        "stop() must not return/raise before the starting transport's own "
        "teardown has actually finished")
    assert _STOP_ARGV in h.ran
    assert ["fake-tailnet", "serve", "status", "--json"] in h.ran, (
        "the teardown was abandoned before it could verify the stop")
    assert h.processes[0].returncode is not None, (
        "the teardown was abandoned before it reaped the subprocess")
    assert h.sockets[0].fileno() == -1
    assert dict(h.policies) == {BASE: "local"}
    assert h.published.owners() == frozenset()
    assert h.manager.running() == {}

    # Cleanup: `start()`'s own outer task is otherwise left with an
    # unretrieved exception (it converts the inner task's cancellation into
    # `TransportError`, matching every other stop-cancels-a-start test in
    # this file).
    with pytest.raises(TransportError, match="still starting"):
        await asyncio.wait_for(starting, 5.0)


@pytest.mark.asyncio
async def test_a_cancellation_racing_an_unrelated_failure_still_wins(
        h, monkeypatch):
    """Minor. `_start_and_unwind`'s except block could finish via whatever
    exception was already in flight (a natural "never announced a
    hostname" timeout, here) while the task itself still carried an
    unhonoured `cancelling() > 0` from a direct cancel delivered while its
    own (protected) teardown for that unrelated failure was still running.
    Matching the other two call sites, the cancellation must win once the
    teardown is done -- the recorded failures are unaffected either way,
    since `stop`'s "still starting" branch reads them from
    `_unwind_failures`, never from which exception this task exits
    through.
    """
    monkeypatch.setattr(manager_mod, "HOSTNAME_TIMEOUT_SECONDS", 0.05)
    loop = asyncio.get_running_loop()
    ran_stop = asyncio.Event()
    release = asyncio.Event()

    def _slow_run(argv):
        h.ran.append(list(argv))
        if argv == _STOP_ARGV:
            loop.call_soon_threadsafe(ran_stop.set)
            deadline = time.monotonic() + 5.0
            while not release.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
        if "--json" in argv:
            return 0, json.dumps(h.status_payload)
        return 0, ""

    monkeypatch.setattr(manager_mod, "_run_argv", _slow_run)
    adapter = _FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV)
    h.register(adapter, lines=[])
    h.stall.add("tailnet")  # never announces -> times out on its own

    outer_task = asyncio.create_task(h.manager.start("tailnet"))
    assert await _settle(lambda: bool(h.spawned)), "the start never spawned"
    inner_task = h.manager._starting["tailnet"]

    # Let the hostname timeout fire on its own -- no `stop()` involved -- so
    # the exception the except block first catches is `TransportError`, not
    # a cancellation.
    assert await _settle(lambda: ran_stop.is_set(), tries=400), (
        "precondition: the natural hostname-timeout unwind must have "
        "reached the (slow) provider stop")

    # Only now, while that unrelated teardown is still in flight, cancel the
    # inner task directly -- absorbed by `_await_uncancellably`, so the
    # teardown itself is undisturbed, but `cancelling()` is left non-zero.
    inner_task.cancel()
    await asyncio.sleep(0)
    assert not inner_task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(inner_task, 5.0)

    assert _STOP_ARGV in h.ran
    assert ["fake-tailnet", "serve", "status", "--json"] in h.ran, (
        "the teardown was abandoned before it could verify the stop")
    assert h.processes[0].returncode is not None, (
        "the teardown was abandoned before it reaped the subprocess")

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(outer_task, 5.0)


@pytest.mark.asyncio
async def test_wait_uncancellably_discards_a_non_cancellation_exception():
    """KI-21, confirmed by reviewers C and B. `_wait_uncancellably`'s own
    docstring promises it never propagates *fut*'s result or exception --
    only whether it is done -- but that promise had no direct test: every
    existing caller's future always ends up cancelled, so the
    `if not fut.cancelled(): fut.exception()` line that discards a
    *non*-cancellation exception has never been exercised by anything in
    this suite. A future second caller passing a future that can end in a
    plain exception would have no test to catch a regression that let it
    leak.

    A bare future, resolved (before this call, deliberately -- the mid-await
    race that can make `_settle_uncancellably` leak such an exception
    straight through `asyncio.shield` is KI-21's separate, still-dormant
    finding, unreachable through this manager's sole caller today, and not
    what this test is pinning) with a `ValueError`, is the direct shape:
    `_wait_uncancellably` must return normally, and must have actually
    *retrieved* the exception -- not merely avoided raising it, which a
    future GC could still turn into an "exception was never retrieved"
    warning. `Future._log_traceback` is asyncio's own bookkeeping for
    exactly that: `set_exception` turns it on, and `exception()` is the one
    thing that turns it back off, so reading it before and after is a
    direct check that the guard line ran, not just that nothing exploded.
    """
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    fut.set_exception(ValueError("boom"))
    assert fut._log_traceback is True, (
        "test setup: a freshly excepted future must start out unretrieved")

    await asyncio.wait_for(manager_mod._wait_uncancellably(fut), 5.0)

    assert fut.cancelled() is False
    assert fut._log_traceback is False, (
        "_wait_uncancellably returned without ever retrieving fut's "
        "exception -- a future GC would log it as 'never retrieved', and "
        "the guard line meant to prevent that never actually ran")
