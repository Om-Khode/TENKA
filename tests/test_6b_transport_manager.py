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
from assistant.io.api.transports.cloudflare import QuickAdapter
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
                 stop_argv: list[str] | None = None) -> None:
        self.name = name
        self._events = events if events is not None else []
        self.refusal = refusal
        self._target_port = target_port
        self._stop_argv = stop_argv
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

    def preflight(self, port: int) -> str | None:
        self._events.append("preflight")
        self.preflight_calls.append(int(port))
        return self.refusal

    def stop_command(self, port: int) -> list[str] | None:
        self.stop_calls.append(int(port))
        return list(self._stop_argv) if self._stop_argv else None


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

    def publish(self, hostname: str, *, owner: str, listener: int) -> None:
        self._events.append("publish")
        super().publish(hostname, owner=owner, listener=listener)

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
        if "status" in argv and "--json" in argv:
            return 0, json.dumps(harness.status_payload)
        return 0, ""

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
    h.registry.register("quick", QuickAdapter())
    h.lines_for["tailnet"] = [b"Available within your tailnet:\n",
                              b"https://box.tail1234.ts.net/\n"]
    h.lines_for["funnel"] = [b"Funnel on:\n",
                             b"https://box.tail1234.ts.net/\n"]
    h.lines_for["quick"] = [b"+---------+\n",
                            b"|  https://ab-cd-ef.trycloudflare.com  |\n"]


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
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter)

    session = await h.manager.start("quick")

    assert h.events == ["preflight", "bind", "register", "spawn", "publish",
                        "serve"], h.events
    assert session.policy_name == "quick"
    assert session.port == port_for("quick", BASE)
    assert session.hostname == "quick.example.com"
    assert session.url == "https://quick.example.com"
    assert adapter.preflight_calls == [port_for("quick", BASE)]
    assert h.policies[session.port] == "quick"
    assert h.published.hosts_for(session.port) == frozenset(
        {"quick.example.com"})
    assert h.served == ["studio-quick"]
    assert sorted(h.manager.running()) == ["quick"]
    assert h.listeners.tasks["quick"] is session.serve_task
    assert h.listeners.sockets["quick"] is session.sock


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
async def test_a_tunnel_that_never_announces_a_hostname_does_not_serve(h):
    """Spec §4: 'a registered socket with no published host is a door with no
    lock on the far side.'

    `TransportSession.hostname` is `str | None`, so this invariant lives in
    the manager rather than in the type (controller Ruling 13a) -- which is
    why it gets its own test asserting the whole teardown.
    """
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter, lines=[b"starting up\n", b"still starting\n"])

    with pytest.raises(TransportError) as excinfo:
        await h.manager.start("quick")

    assert "hostname" in str(excinfo.value)
    assert h.spawned, "precondition: the tunnel really was spawned"
    assert "publish" not in h.events, "a nameless session reached publish()"
    assert h.served == [], "a nameless session was served"
    assert port_for("quick", BASE) not in h.policies
    assert h.published.owners() == frozenset()
    assert h.sockets[0].fileno() == -1, "the socket was not closed"
    assert h.processes[0].returncode is not None, "the subprocess was not reaped"
    assert h.manager.running() == {}
    assert "quick" not in h.listeners.tasks
    assert "quick" not in h.listeners.sockets


@pytest.mark.asyncio
async def test_a_tunnel_that_stalls_past_the_timeout_does_not_serve(h,
                                                                   monkeypatch):
    """The other half of 'never announces': a process that says nothing at
    all and never closes its output. Bounded by `HOSTNAME_TIMEOUT_SECONDS`,
    shortened here so the test costs milliseconds instead of thirty seconds.
    """
    monkeypatch.setattr(manager_mod, "HOSTNAME_TIMEOUT_SECONDS", 0.05)
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter, lines=[])
    h.stall.add("quick")

    with pytest.raises(TransportError):
        await h.manager.start("quick")

    assert h.served == []
    assert port_for("quick", BASE) not in h.policies
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
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter, lines=[b"x" * 50 + b"\n" for _ in range(100)])

    with pytest.raises(TransportError):
        await h.manager.start("quick")

    proc = h.processes[0]
    assert proc.stdout.reads < 100, (
        f"scanned {proc.stdout.reads} lines -- the byte bound did not hold")


@pytest.mark.asyncio
async def test_starting_a_transport_twice_is_refused(h):
    """And the refusal must not touch the session already running -- an
    'already running' path that ran the teardown would take the live tunnel
    down for the sake of the second caller's error message."""
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter)

    session = await h.manager.start("quick")

    with pytest.raises(TransportError, match="already running"):
        await h.manager.start("quick")

    assert len(h.spawned) == 1
    assert sorted(h.manager.running()) == ["quick"]
    assert h.manager.running()["quick"].owner == session.owner
    assert session.sock.fileno() != -1, "the refusal closed the live socket"
    assert h.published.hosts_for(session.port) == frozenset(
        {"quick.example.com"})


@pytest.mark.asyncio
async def test_starting_an_unknown_transport_is_refused(h):
    with pytest.raises(TransportError, match="nonesuch"):
        await h.manager.start("nonesuch")
    assert h.spawned == []
    assert h.sockets == []


# ═════════════════════════════════════════════════════════════════════════
# KI-17 layer 1 -- against the real argument vector
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["tailnet", "funnel", "quick"])
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
    honest = _FakeAdapter("quick", events=h.events)
    h.register(honest)
    await h.manager.start("quick")
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

    adapter = _StderrAdapter("quick", events=h.events)
    h.register(adapter)

    session = await h.manager.start("quick")
    try:
        assert session.hostname == "stderr.example.com"
        assert h.published.hosts_for(session.port) == frozenset(
            {"stderr.example.com"})
    finally:
        await h.manager.stop("quick")

    assert session.process.returncode is not None, (
        "the real subprocess was not reaped by the teardown")


# ═════════════════════════════════════════════════════════════════════════
# Stop
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_stop_unpublishes_deregisters_drops_raises_and_closes_the_socket(h):
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter)
    session = await h.manager.start("quick")
    h.raises.grant("dev-1", "quick", frozenset({Capability.OBSERVE}), 60,
                   "test", "why")

    # Preconditions, so the assertions below cannot pass against a manager
    # whose start never did anything in the first place.
    assert h.published.hosts_for(session.port) == frozenset(
        {"quick.example.com"})
    assert h.policies[session.port] == "quick"
    assert h.raises.capabilities_for("dev-1", "quick") == frozenset(
        {Capability.OBSERVE})
    assert session.sock.fileno() != -1
    assert not session.serve_task.done()

    await h.manager.stop("quick")

    assert h.published.hosts_for(session.port) == frozenset()
    assert h.published.owners() == frozenset()
    assert session.port not in h.policies
    assert h.raises.capabilities_for("dev-1", "quick") == frozenset()
    assert session.serve_task.done()
    assert session.sock.fileno() == -1
    assert session.process.returncode is not None
    assert h.manager.running() == {}
    assert "quick" not in h.listeners.tasks
    assert "quick" not in h.listeners.sockets


@pytest.mark.asyncio
async def test_stop_is_safe_to_run_twice(h):
    """A crash handler and an orderly shutdown both call the teardown, and
    both may call it twice."""
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter)
    session = await h.manager.start("quick")

    await h.manager.stop("quick")
    assert session.sock.fileno() == -1, (
        "precondition: the first stop must actually have torn down")

    await h.manager.stop("quick")

    assert h.manager.running() == {}
    assert session.port not in h.policies
    assert h.published.owners() == frozenset()


@pytest.mark.asyncio
async def test_stopping_a_transport_that_never_started_is_not_an_error(h):
    await h.manager.stop("quick")
    assert h.manager.running() == {}


@pytest.mark.asyncio
async def test_a_crashed_tunnel_runs_the_same_teardown_as_an_orderly_stop(h):
    """Asserted as an equality between two runs of one transport in one
    harness -- the crash first, the orderly stop second -- so 'the same
    teardown' is a comparison rather than two lists of assertions that could
    drift apart.
    """
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter)

    crashed = await h.manager.start("quick")
    h.raises.grant("dev-1", "quick", frozenset({Capability.OBSERVE}), 60,
                   "test", "why")
    assert h.raises.capabilities_for("dev-1", "quick")  # precondition
    crashed.process.exit(1)
    # The session stops being reported as running the instant the teardown
    # takes it, so the wait is for the *last* observable step -- otherwise
    # the effects below would be read halfway through the unwind.
    assert await _settle(lambda: not h.manager.running()
                         and crashed.sock.fileno() == -1), (
        "the crashed tunnel's listener was left running")
    after_crash = _effects(h, "quick", crashed.port, crashed.sock,
                           crashed.process, crashed.serve_task, "dev-1")

    h.lines_for["quick"] = [b"https://second.example.com\n"]
    orderly = await h.manager.start("quick")
    h.raises.grant("dev-1", "quick", frozenset({Capability.OBSERVE}), 60,
                   "test", "why")
    await h.manager.stop("quick")
    after_stop = _effects(h, "quick", orderly.port, orderly.sock,
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
    adapter = _FakeAdapter("quick", events=h.events)
    h.register(adapter)

    first = await h.manager.start("quick")
    await h.manager.stop("quick")

    h.lines_for["quick"] = [b"https://second.example.com\n"]
    second = await h.manager.start("quick")

    assert first.owner != second.owner
    assert first.owner != "quick" and second.owner != "quick"
    assert h.published.hosts_for(second.port) == frozenset(
        {"second.example.com"}), "the first run's hostname is still trusted"

    await h.manager.stop("quick")
    assert h.published.hosts_for(second.port) == frozenset()


@pytest.mark.asyncio
async def test_stop_all_leaves_only_the_local_listener(h):
    for name in ("tailnet", "funnel", "quick"):
        h.register(_FakeAdapter(name, events=h.events))
        await h.manager.start(name)

    assert sorted(h.manager.running()) == ["funnel", "quick", "tailnet"]
    assert len(h.policies) == 4

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
    adapter = _FakeAdapter("quick", events=h.events, stop_argv=None)
    h.register(adapter)
    session = await h.manager.start("quick")

    await h.manager.stop("quick")

    assert h.ran == [], f"a foreground transport ran {h.ran}"
    assert session.process.returncode is not None
    assert session.process.terminated == 1


@pytest.mark.asyncio
async def test_stop_all_reports_a_failed_verification_without_stranding_others(h):
    """A shutdown must not be blocked by one transport that will not stop --
    but the failure must still be reported, not swallowed."""
    h.register(_FakeAdapter("tailnet", events=h.events, stop_argv=_STOP_ARGV))
    h.register(_FakeAdapter("quick", events=h.events))
    tailnet = await h.manager.start("tailnet")
    quick = await h.manager.start("quick")
    h.status_payload = {"Web": {"box.ts.net:9443": {}}}

    failures = await h.manager.stop_all()

    assert len(failures) == 1 and "9443" in failures[0]
    assert h.manager.running() == {}
    assert quick.sock.fileno() == -1, "the healthy transport was stranded"
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
    port = port_for("quick", BASE)
    sock = server_mod.bind_listener(port)
    h.sockets.append(sock)
    task = _sleeping_task("quick")
    h.serve_tasks.append(task)
    nameless = TransportSession(policy_name="quick", port=port, owner="o",
                                process=_FakeProcess([]), sock=sock,
                                serve_task=task, hostname=None)
    named = replace(nameless, hostname="quick.example.com")

    assert is_serving(nameless) is False
    assert is_serving(named) is True, (
        "precondition: a session that *did* announce must be serving, or the "
        "assertion above holds for the wrong reason")

    with pytest.raises(TransportError):
        require_serving(nameless)
    assert require_serving(named) is named

    h.manager._sessions["quick"] = nameless
    assert h.manager.running() == {}, (
        "a nameless session was reported as a running transport")
