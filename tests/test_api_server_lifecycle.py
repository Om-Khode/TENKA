"""The daemon is off by default and binds loopback when on."""
import pathlib

import pytest


def test_the_flag_defaults_to_off():
    from assistant.core import runtime_config
    import assistant.config  # noqa: F401  — registers the settings
    assert runtime_config.REGISTRY["studio_api_enabled"]["default"] is False


def test_the_flag_is_documented():
    from assistant.core import runtime_config
    import assistant.config  # noqa: F401
    assert runtime_config.REGISTRY["studio_api_enabled"]["description"].strip()


def test_the_server_binds_loopback_only():
    source = pathlib.Path("assistant/io/api/server.py").read_text(encoding="utf-8")
    assert '"127.0.0.1"' in source
    assert '"0.0.0.0"' not in source, "the daemon must not bind every interface"


def test_main_starts_it_only_behind_the_flag():
    source = pathlib.Path("assistant/main.py").read_text(encoding="utf-8")
    assert "STUDIO_API_ENABLED" in source, "main.py does not consult the flag"
    guard_line = [line for line in source.splitlines() if "STUDIO_API_ENABLED" in line]
    assert any("if" in line for line in guard_line), "the flag is read but not guarded on"


def test_a_normal_shutdown_does_not_call_the_kill_switch():
    """Reversed from the original version of this test, which required
    server.shutdown() (the kill switch that rotates the instance secret via
    vault.reset()) to fire unconditionally at shutdown. That was backwards:
    this block runs on every *orderly* exit -- Ctrl+C, the "shutdown" voice
    intent, an exception unwinding through the surrounding `finally` -- so
    calling the kill switch there meant a phone paired yesterday had to be
    re-paired after every ordinary restart, and the token meant to print
    once, at first pairing, printed on every boot. Only `_stop_studio_daemon`
    (stop serving, never touch the vault) belongs in this block now;
    server.shutdown/shutdown_studio_api must not appear in it at all.
    """
    source = pathlib.Path("assistant/main.py").read_text(encoding="utf-8")
    marker = "# ─── Studio daemon shutdown"
    idx = source.index(marker)
    # Up to the next top-level section marker (or a generous window if this
    # is the last one in the function) -- wide enough to catch the call
    # wherever in the block it might be reintroduced, narrow enough not to
    # accidentally match the unrelated bridge-shutdown code right after it.
    block = source[idx: idx + 2_500]
    assert "shutdown_studio_api(_studio_task, _studio_vault)" not in block, (
        "the kill switch must not be called from the normal shutdown path"
    )
    assert "_stop_studio_daemon(_studio_task)" in block, (
        "normal shutdown must still stop serving"
    )


def test_the_kill_switch_itself_is_still_fully_defined_and_reachable():
    """The fix above removes the *automatic* call, not the mechanism. Nothing
    in this milestone gives server.shutdown() a deliberate trigger yet (a
    future "revoke every Studio device" admin action would call it
    explicitly) -- it stays covered by test_api_hardening.py's
    test_shutdown_revokes_every_device and this file's own
    test_shutdown_revokes_devices_and_eventually_frees_the_port.
    """
    from assistant.io.api import server
    assert callable(server.shutdown)


@pytest.mark.asyncio
async def test_serve_returns_a_task_that_can_be_cancelled(tmp_path):
    import asyncio
    from assistant.io.api import server
    from assistant.io.api.vault import TokenVault
    from tests.fakes.studio_runtime import build_fake_runtime

    task = server.serve(build_fake_runtime(), TokenVault(tmp_path),
                        host="127.0.0.1", port=8931, origins=["http://localhost:3000"])
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert task.done()


@pytest.mark.asyncio
async def test_cancelling_the_serve_task_actually_releases_the_port(tmp_path):
    """A kill switch that revokes every token but leaves uvicorn's listening
    socket open has only paused the daemon at the application layer -- the
    OS still shows the port bound. `uvicorn.Server.main_loop()` awaits
    `asyncio.sleep(0.1)` with no try/finally around it, so a bare
    `task.cancel()` interrupts that sleep and unwinds straight out of
    `_serve()`, skipping the `await self.shutdown(sockets=...)` call that
    closes `self.servers` -- proven here by rebinding the same host:port
    immediately after cancellation and requiring it to succeed.
    """
    import asyncio
    import socket

    from assistant.io.api import server
    from assistant.io.api.vault import TokenVault
    from tests.fakes.studio_runtime import build_fake_runtime

    port = 8932
    task = server.serve(build_fake_runtime(), TokenVault(tmp_path),
                        host="127.0.0.1", port=port, origins=["http://localhost:3000"])
    await asyncio.sleep(0.2)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.1)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        pytest.fail(f"port {port} is still bound after cancellation: {exc}")
    finally:
        probe.close()


@pytest.mark.asyncio
async def test_shutdown_revokes_devices_and_eventually_frees_the_port(tmp_path):
    """Drives `server.shutdown()` itself -- the entry point callers actually
    use -- on a real started daemon, not a hand-rolled cancel-and-await that
    only mimics it. Two halves, proven in the order `shutdown()`'s own
    docstring documents as required:

    Device revocation is synchronous: `vault.reset()` runs inline inside
    `shutdown()`, so it holds the instant the call returns, no `await`
    needed. Port release is not: `shutdown()` only calls `task.cancel()`,
    which *schedules* cancellation -- `_run()`'s own cleanup (the
    `await server.shutdown()` that actually closes the listening socket)
    still has to run on the event loop. Checked directly: calling
    `server.shutdown()` and probing the port with no intervening `await` at
    all shows it still bound (confirmed manually while writing this test,
    not asserted here as a permanent regression case, since a test that
    must observe a race is not one this suite should depend on timing to
    pass). What this test proves is the guarantee that actually matters and
    that `main.py` actually relies on: after giving the task the one
    `await` its own cancellation needs, both halves hold.
    """
    import asyncio
    import contextlib
    import socket

    from assistant.io.api import server
    from assistant.io.api.vault import Capability, TokenVault
    from tests.fakes.studio_runtime import build_fake_runtime

    port = 8933
    vault = TokenVault(tmp_path)
    vault.issue("studio", frozenset(Capability))
    task = server.serve(build_fake_runtime(), vault, host="127.0.0.1", port=port,
                        origins=["http://localhost:3000"])
    await asyncio.sleep(0.2)

    server.shutdown(task, vault)

    # Half one: revocation, synchronous, already true.
    assert vault.devices() == []

    # Half two: port release, scheduled by shutdown()'s task.cancel() but
    # not yet run -- give the task the turn its own cleanup needs.
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        pytest.fail(f"port {port} is still bound after shutdown(): {exc}")
    finally:
        probe.close()


def test_the_exporter_writes_a_schema(tmp_path):
    import json
    import subprocess
    import sys

    out = tmp_path / "openapi.json"
    result = subprocess.run([sys.executable, "tools/export_openapi.py", str(out)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert "/v1/status" in schema["paths"]
    # GET /v1/memory/{scope} was split into three static routes (review
    # finding, 2026-08-08): one dynamic-scope route describing its response
    # as a three-way union gave a generated client no discriminator to key
    # off; three routes give a clean 1:1 type each. The URLs themselves are
    # unchanged -- these three were always the only ones a client called.
    assert "/v1/memory/knowledge" in schema["paths"]
    assert "/v1/memory/preferences" in schema["paths"]
    assert "/v1/memory/procedures" in schema["paths"]


# ─── _StudioDispatch — the only path from a request into the pipeline ─────
# Importing assistant.main only loads its module (functions, classes,
# logging setup) -- it never starts the assistant. Nothing here calls
# main() or async_main(), so no microphone, no desktop control, no API
# spend.
#
# `submit()` takes a grant set positionally with no default (Milestone 6a.5).
# These tests are about the busy/queue mechanics, not about the gate, so they
# hand over the full set -- but they have to hand over *something*, which is
# the point of the missing default.
from assistant.core.capabilities import Capability

_ALL_GRANTS = frozenset(Capability)


@pytest.mark.asyncio
async def test_studio_dispatch_puts_the_shape_the_consumer_loop_expects():
    """main.py's queue consumer does `source, text = item[0], item[1]`, the
    same first two slots the existing "chat" source uses.

    The third slot is the studio turn's grant set (Milestone 6a.5) and the
    fourth is its principal (6b, KI-13): both have to travel with the turn,
    because the turn runs later on the consumer's task and the request that
    authorised them is gone by then. Local sources still enqueue 2-tuples (or
    a 3-tuple whose third slot is stt_ms); `_grants_for_item` and
    `_principal_for_item` are what tell the two apart, by source."""
    import assistant.main as m

    while True:
        try:
            m._input_queue.get_nowait()
        except Exception:
            break

    dispatch = m._StudioDispatch()
    turn_id, session_id, accepted, reason = await dispatch.submit("hello studio", _ALL_GRANTS, "device:probe")
    assert (accepted, reason) == (True, "")
    assert turn_id == "studio-1"
    assert isinstance(session_id, str)  # get_current_session_id()'s real return type

    item = m._input_queue.get_nowait()
    assert item == ("studio", "hello studio", _ALL_GRANTS, "device:probe")


@pytest.mark.asyncio
async def test_studio_dispatch_refuses_rather_than_queues_a_concurrent_submit():
    """The single-turn state: two real concurrent submits (asyncio.gather
    through the actual dispatch, not a hand-held lock standing in for one)
    must produce exactly one acceptance and one refusal. The original
    version of this test manually held `dispatch._lock` across an
    `asyncio.sleep(0.2)` to *simulate* a turn in flight -- a state production
    code never produced, since submit() itself never awaited between
    checking and acquiring the lock. There is no lock left to hold: busy is
    a plain flag now, set the instant a submit is accepted and cleared only
    by mark_done() (called from process_text_from_queue's own `finally`
    once a "studio"-sourced turn genuinely finishes) -- proven below by
    checking it is still refused *before* mark_done() runs, and accepted
    again only *after*.
    """
    import asyncio

    import assistant.main as m

    dispatch = m._StudioDispatch()

    results = await asyncio.gather(
        dispatch.submit("first", _ALL_GRANTS, "device:probe"), dispatch.submit("second", _ALL_GRANTS, "device:probe"),
    )
    accepted = [r for r in results if r[2] is True]
    refused = [r for r in results if r[2] is False]
    assert len(accepted) == 1, f"expected exactly one acceptance, got {results}"
    assert len(refused) == 1, f"expected exactly one refusal, got {results}"
    assert refused[0] == ("", "", False, "busy")

    # Still busy: the accepted turn has not finished (mark_done() has not
    # run) -- a third submit must still be refused.
    still_busy = await dispatch.submit("third", _ALL_GRANTS, "device:probe")
    assert still_busy[2] is False

    dispatch.mark_done()
    now_free = await dispatch.submit("fourth", _ALL_GRANTS, "device:probe")
    assert now_free[2] is True


@pytest.mark.asyncio
async def test_mark_done_is_a_no_op_when_already_idle():
    import assistant.main as m

    dispatch = m._StudioDispatch()
    dispatch.mark_done()  # must not raise
    assert dispatch.busy is False


def test_process_text_from_queue_is_wired_to_clear_busy_for_a_studio_turn():
    """process_text_from_queue is a heavy, real pipeline (intent
    classification, LLM calls, TTS) -- exercising it end-to-end from a unit
    test would mean the exact live-runtime/API-spend/audio calls this test
    suite must not make. Checked structurally instead, the same way this
    file already pins the startup/shutdown flag-guards: mark_done() must be
    called from within the function's own `finally` block, gated on
    `source == "studio"`, not unconditionally (a voice/chat turn never went
    through _StudioDispatch.submit() in the first place, so it must not
    clear a flag it never set).
    """
    source = pathlib.Path("assistant/main.py").read_text(encoding="utf-8")
    fn_start = source.index("async def process_text_from_queue(")
    finally_idx = source.index("\n    finally:", fn_start)
    # Bounded by the end of the function, not by a character count: the block
    # is the last one in it, and the next top-level `async def` is
    # _process_one_queued_item -- which contains its own mark_done() call and
    # would satisfy this assertion for the wrong reason if the window ran on.
    # (A fixed window instead breaks on comment edits, which is what it did
    # when the status bracket landed beside the call.)
    fn_end = source.index("\nasync def ", finally_idx)
    finally_block = source[finally_idx:fn_end]
    assert "_studio_dispatch.mark_done()" in finally_block, (
        "process_text_from_queue's finally block does not clear studio's busy flag"
    )
    assert 'source == "studio"' in finally_block, (
        "mark_done() must be gated on source == \"studio\", not unconditional"
    )


# ─── fix wave 2: the busy flag must not strand permanently ────────────────
# process_text_from_queue does three things before its *own* try: block --
# construct a TurnTracker, register it with _telemetry.set_current_tracker,
# and (if a wake listener is running) pause it. A raise in any of those
# skipped that function's own finally entirely, leaving _busy permanently
# True: no watchdog, no timeout, the Studio channel answering 409 to every
# message until the process restarted. Driven through the actual production
# seam, _process_one_queued_item (what the consumer loop now calls), not
# through process_text_from_queue directly -- the fix lives in that seam's
# own try/finally, not inside process_text_from_queue.
@pytest.mark.asyncio
async def test_a_raise_before_process_text_from_queues_own_try_still_clears_busy(monkeypatch):
    import assistant.main as m

    dispatch = m._StudioDispatch()
    monkeypatch.setattr(m, "_studio_dispatch", dispatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("telemetry is down")

    # TurnTracker construction is the very first line of
    # process_text_from_queue -- before session_mod, before bridge, before
    # its own try:. Forcing the raise here hits the earliest possible point
    # in the "skips the finally entirely" window the review named.
    monkeypatch.setattr(m._telemetry, "TurnTracker", _boom)

    turn_id, _, accepted, _ = await dispatch.submit("first", _ALL_GRANTS, "device:probe")
    assert accepted is True
    assert dispatch.busy is True

    with pytest.raises(RuntimeError):
        await m._process_one_queued_item(("studio", "first"), bridge=None)

    assert dispatch.busy is False, (
        "a raise before process_text_from_queue's own try: left the busy flag stranded"
    )

    second = await dispatch.submit("second", _ALL_GRANTS, "device:probe")
    assert second[2] is True, (
        "the studio channel is still refusing after the raise -- exactly the "
        "permanent lockout the review flagged"
    )


@pytest.mark.asyncio
async def test_a_normal_turn_through_the_consumer_seam_still_clears_busy(monkeypatch):
    """The mirror of the test above: the new outer try/finally must not have
    simply disabled the busy mechanism just built. process_text_from_queue
    itself is stubbed (it is the real pipeline -- intent, LLM, TTS -- which
    this suite must not run), so this proves the *seam's* plumbing, not the
    pipeline's correctness.
    """
    import assistant.main as m

    dispatch = m._StudioDispatch()
    monkeypatch.setattr(m, "_studio_dispatch", dispatch)

    async def _fake_process(source, text, bridge, stt_ms=None, grants=None,
                            principal=None):
        return None

    monkeypatch.setattr(m, "process_text_from_queue", _fake_process)

    await dispatch.submit("hello", _ALL_GRANTS, "device:probe")
    assert dispatch.busy is True

    await m._process_one_queued_item(("studio", "hello"), bridge=None)

    assert dispatch.busy is False


@pytest.mark.asyncio
async def test_the_seam_does_not_clear_busy_before_the_turn_actually_finishes(monkeypatch):
    """The other half of "did not simply disable the mechanism": while
    process_text_from_queue is still running, a second submit must still be
    refused -- the outer finally must fire only after the call returns (or
    raises), never eagerly."""
    import asyncio

    import assistant.main as m

    dispatch = m._StudioDispatch()
    monkeypatch.setattr(m, "_studio_dispatch", dispatch)

    release = asyncio.Event()

    async def _slow_process(source, text, bridge, stt_ms=None, grants=None,
                            principal=None):
        await release.wait()

    monkeypatch.setattr(m, "process_text_from_queue", _slow_process)

    await dispatch.submit("hello", _ALL_GRANTS, "device:probe")
    task = asyncio.create_task(m._process_one_queued_item(("studio", "hello"), bridge=None))
    await asyncio.sleep(0)  # let the task actually start awaiting release

    still_busy = await dispatch.submit("second", _ALL_GRANTS, "device:probe")
    assert still_busy[2] is False, "the seam cleared busy before the turn actually finished"

    release.set()
    await task
    assert dispatch.busy is False


@pytest.mark.asyncio
async def test_studio_dispatch_abort_reaches_the_shared_abort_controller():
    import assistant.main as m
    from assistant.core.abort import abort

    dispatch = m._StudioDispatch()
    try:
        assert await dispatch.abort() is True
        assert abort.is_aborted() is True
        assert abort.reason == "studio"
    finally:
        abort.reset()


# ─── _start_studio_daemon / _stop_studio_daemon ────────────────────────────
# Split out of async_main() so these exact sequences are drivable without
# booting the assistant. serve() is always monkeypatched below -- never the
# real one -- so build_studio_runtime()'s Live* runtimes are constructed
# (side-effect-free: none of them touch storage in __init__, only in their
# async methods, which nothing here calls) but never handed to a real
# uvicorn app or exercised.


@pytest.mark.asyncio
async def test_a_failed_start_does_not_leak_a_status_subscription(tmp_path, monkeypatch):
    """StatusBroadcaster.subscribe() has no unsubscribe. A daemon that fails
    to start (e.g. create_app()'s eager instance_secret() check raising on a
    bad TENKA_SECRET) must not have already subscribed its hub -- otherwise
    every future status.set() call, which fires on every phase transition,
    keeps appending to a hub nothing will ever drain, for the rest of the
    process, on a machine where the daemon is off.

    Also asserts the pair-store half of this exact scenario (Task 11 review,
    round 2): this test already drove a synchronous serve() failure and
    asserted only on the subscription, which is precisely what let
    `_studio_pair_store` stay orphaned -- assigned before serve() was ever
    called, never cleared in the `except` -- pass silently for a full round.
    One scenario, both failure modes, pinned together so that does not
    happen again.
    """
    import assistant.config as config
    import assistant.main as m
    from assistant.io.status_broadcaster import status

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    def _raise(*args, **kwargs):
        raise ValueError("bad TENKA_SECRET")

    monkeypatch.setattr("assistant.io.api.server.serve", _raise)

    before = len(status._subscribers)
    result = await m._start_studio_daemon()
    assert result is None
    assert len(status._subscribers) == before, (
        "a failed serve() must leave no trace on the broadcaster's subscriber list"
    )
    assert m._studio_pair_store is None, (
        "a serve() that raised synchronously must not leave a pair store "
        "behind -- /studio pair would keep minting unredeemable codes for "
        "a daemon that never actually started"
    )


@pytest.mark.asyncio
async def test_a_failed_start_leaves_studio_pair_refusing(tmp_path, monkeypatch):
    """The behavioural twin of the test above, driven through the actual
    slash command rather than the raw global -- proved by execution, per
    the review: patch serve() to raise the same synchronous
    `create_app()`-eager-check failure, start the daemon, then run
    `/studio pair` and assert it refuses (message *and* no code-shaped
    string), not that it silently succeeds against an orphaned store.
    """
    import re

    import assistant.config as config
    import assistant.main as m
    from assistant import slash_commands

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    def _raise(*args, **kwargs):
        raise ValueError("bad TENKA_SECRET")

    monkeypatch.setattr("assistant.io.api.server.serve", _raise)

    result = await m._start_studio_daemon()
    assert result is None, "sanity: the start must actually have failed"

    response = slash_commands.handle("/studio pair testphone")

    assert "not running" in response.lower()
    assert not re.search(r"[0-9A-Z]{4}-[0-9A-Z]{4}", response), (
        "a failed start must not leave a code-minting store reachable"
    )


@pytest.mark.asyncio
async def test_a_successful_start_subscribes_exactly_once(tmp_path, monkeypatch):
    """The mirror of the test above: once serve() *has* returned a task, the
    subscription must actually happen -- ordering the fix around "subscribe
    only after serve() succeeds" must not turn into "never subscribe"."""
    import asyncio

    import assistant.config as config
    import assistant.main as m
    from assistant.io.status_broadcaster import status

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    async def _noop() -> None:
        return None

    def _fake_serve(*args, **kwargs):
        return asyncio.create_task(_noop())

    monkeypatch.setattr("assistant.io.api.server.serve", _fake_serve)

    before = len(status._subscribers)
    task = await m._start_studio_daemon()
    try:
        assert task is not None
        assert len(status._subscribers) == before + 1
    finally:
        # This subscription is real and permanent (no unsubscribe exists),
        # so this test's own hub must never actually publish anything for
        # the rest of the session: drop the reference and let the task
        # finish on its own rather than leaving anything running.
        await task


@pytest.mark.asyncio
async def test_a_successful_start_subscribes_publish_status_not_publish(tmp_path, monkeypatch):
    """The precise regression the socket-contract fix repairs: reverting
    main.py back to `status.subscribe(_studio_hub.publish)` would still
    subscribe *something* (the test above stays green), still type-check,
    and still pass lint-imports -- but every status frame reaching a
    browser would be snake_case again. Pin the *method identity* of what
    got subscribed, not just that a subscription happened. Drives the real
    `_start_studio_daemon()` (serve() faked, same as the sibling test above)
    rather than statically parsing main.py's source, matching this file's
    existing pattern for exercising that function."""
    import asyncio

    import assistant.config as config
    import assistant.main as m
    from assistant.io.api.events import EventHub
    from assistant.io.status_broadcaster import status

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    async def _noop() -> None:
        return None

    def _fake_serve(*args, **kwargs):
        return asyncio.create_task(_noop())

    monkeypatch.setattr("assistant.io.api.server.serve", _fake_serve)

    task = await m._start_studio_daemon()
    try:
        assert task is not None
        subscribed = status._subscribers[-1]
        assert subscribed.__func__ is EventHub.publish_status, (
            "main.py must subscribe publish_status (translates a "
            "broadcaster event before publishing it), not publish (raw "
            f"passthrough) -- got {subscribed.__func__!r}"
        )
    finally:
        await task


@pytest.mark.asyncio
async def test_stop_studio_daemon_tolerates_no_task():
    import assistant.main as m
    await m._stop_studio_daemon(None)  # must not raise


@pytest.mark.asyncio
async def test_stop_studio_daemon_cancels_a_running_task():
    import asyncio

    import assistant.main as m

    async def _run_forever():
        await asyncio.sleep(100)

    task = asyncio.create_task(_run_forever())
    await asyncio.sleep(0)  # let it actually start running
    await m._stop_studio_daemon(task)
    assert task.done()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_stop_studio_daemon_retrieves_a_crashed_tasks_exception(caplog):
    """A task that already finished before shutdown got here (e.g. the port
    was taken, so _run() raised inside the task rather than at the
    synchronous serve() call _start_studio_daemon() already guards) must
    have its exception retrieved and logged -- not silently skipped, which
    is what `if not task.done():` alone did before this fix, leaving
    asyncio's own unretrieved-exception warning to fire later, unredacted."""
    import asyncio
    import contextlib
    import logging

    import assistant.main as m

    async def _boom():
        raise RuntimeError("port already in use: 8787")

    task = asyncio.create_task(_boom())
    with contextlib.suppress(Exception):
        await task
    assert task.done()

    with caplog.at_level(logging.WARNING, logger="main"):
        await m._stop_studio_daemon(task)

    assert any("Studio daemon exited early" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_successful_start_leaves_the_vault_reachable_for_shutdown(tmp_path, monkeypatch):
    """The kill switch (assistant.io.api.server.shutdown) needs the same
    TokenVault _start_studio_daemon() built, but that object is a local
    inside this function -- it never rides the returned task. main.py's
    shutdown site reads module-level `_studio_vault` instead; this pins
    that the module-level name is actually set to a real, usable vault
    after a successful start, not left None or stale from a prior test.
    """
    import asyncio

    import assistant.config as config
    import assistant.main as m
    from assistant.io.api.vault import TokenVault

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    async def _noop() -> None:
        return None

    def _fake_serve(*args, **kwargs):
        return asyncio.create_task(_noop())

    monkeypatch.setattr("assistant.io.api.server.serve", _fake_serve)

    task = await m._start_studio_daemon()
    try:
        assert isinstance(m._studio_vault, TokenVault)
        assert m._studio_vault.devices(), "the studio vault should hold its issued device"
    finally:
        await task


@pytest.mark.asyncio
async def test_a_successful_start_leaves_the_dispatch_reachable_too(tmp_path, monkeypatch):
    """process_text_from_queue's finally block reads module-level
    `_studio_dispatch` the same way main.py's shutdown site reads
    `_studio_vault` -- both are locals inside _start_studio_daemon() until
    this pins that the module-level name actually gets set."""
    import asyncio

    import assistant.config as config
    import assistant.main as m

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    async def _noop() -> None:
        return None

    def _fake_serve(*args, **kwargs):
        return asyncio.create_task(_noop())

    monkeypatch.setattr("assistant.io.api.server.serve", _fake_serve)

    task = await m._start_studio_daemon()
    try:
        assert isinstance(m._studio_dispatch, m._StudioDispatch)
        assert m._studio_dispatch.busy is False
    finally:
        await task


@pytest.mark.asyncio
async def test_a_started_daemons_token_never_reaches_a_log_record(tmp_path, monkeypatch, caplog):
    """Verified structurally, not just via redaction surviving it: the raw
    token must never be handed to `logger` at all. redact_secrets() catches
    it via the bare high-entropy-token path if it ever is, but that path is
    probabilistic (needs a digit plus mixed case or a separator), and
    DEBUG_LOG defaults to true on a fresh install -- so a log line that
    merely relies on redaction working is one heuristic away from KI-12."""
    import asyncio
    import logging

    import assistant.config as config
    import assistant.main as m

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    async def _noop() -> None:
        return None

    def _fake_serve(*args, **kwargs):
        return asyncio.create_task(_noop())

    monkeypatch.setattr("assistant.io.api.server.serve", _fake_serve)

    printed: list[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)),
    )

    with caplog.at_level(logging.INFO, logger="main"):
        task = await m._start_studio_daemon()
    assert task is not None
    await task

    token_line = next(line for line in printed if "TENKA Studio token" in line)
    raw_token = token_line.rsplit(":", 1)[-1].strip()
    assert len(raw_token) > 20, "sanity: the console line must carry the real token"

    for record in caplog.records:
        assert raw_token not in record.getMessage()
        assert raw_token not in str(record.msg)


# ─── Fix round (Task 11 review): the pair store must be the routes' own ───
# C1/Fix 4: nothing pinned that the module global `/studio pair` reads is
# the *same object* serve_studio_api() (main.py's alias for
# assistant.io.api.server.serve()) hands to create_app() as `pair_store=`.
# A wrong object passed to that kwarg would have gone uncaught -- every
# `/studio pair` test drives the global directly via a fixture, never
# through a real _start_studio_daemon() call.


@pytest.mark.asyncio
async def test_a_successful_start_leaves_a_pair_store_the_routes_actually_hold(
    tmp_path, monkeypatch,
):
    """The mirror of test_a_successful_start_leaves_the_vault_reachable_for_
    shutdown above, for the pair store: a start must leave
    `assistant.main._studio_pair_store` set to the *exact* `PairCodeStore`
    instance handed to `serve()` as `pair_store=` -- not merely *a*
    `PairCodeStore`, which a bug swapping in a second, freshly-constructed
    one would still satisfy.
    """
    import asyncio

    import assistant.config as config
    import assistant.main as m
    from assistant.io.api.pairing import PairCodeStore

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    captured: dict = {}

    async def _noop() -> None:
        return None

    def _fake_serve(*args, **kwargs):
        captured.update(kwargs)
        return asyncio.create_task(_noop())

    monkeypatch.setattr("assistant.io.api.server.serve", _fake_serve)

    task = await m._start_studio_daemon()
    try:
        assert isinstance(m._studio_pair_store, PairCodeStore)
        assert "pair_store" in captured, (
            "_start_studio_daemon() never passed pair_store= to serve() at all"
        )
        assert captured["pair_store"] is m._studio_pair_store, (
            "the module global is not the same store handed to serve() -- "
            "/studio pair would mint into an object the real app's routes "
            "never see"
        )
    finally:
        await task


# ─── the UI bundle reaches the running daemon ──────────────────────────────
# Task 7 built `UiBundle` and `mount_ui()` and left the path decision to
# main.py, because `io/api` may not import `config`. Task 16 built the
# artefact. Between the two, nothing actually resolved a bundle at startup:
# every `mount_ui` call site outside the tests passed `None`, `studio_ui_path`
# was a setting nothing read, and the daemon served no `/` at all while every
# unit test in the milestone stayed green. These are the tests that would have
# caught that, so they assert the wiring rather than the loader.


def _fake_serve_capturing(captured: dict):
    import asyncio

    async def _noop() -> None:
        return None

    def _serve(*args, **kwargs):
        captured.update(kwargs)
        return asyncio.create_task(_noop())

    return _serve


@pytest.mark.asyncio
async def test_startup_serves_a_local_export_when_studio_ui_path_is_set(
    tmp_path, monkeypatch,
):
    """The developer override. It wins over the vendored zip so that iterating
    on Studio needs no re-packaging."""
    import assistant.config as config
    import assistant.main as m
    from tests.fakes.studio_ui import write_ui_dir

    export = write_ui_dir(tmp_path / "out", "does-not-have-to-match")
    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    monkeypatch.setattr(config, "STUDIO_UI_PATH", str(export))

    captured: dict = {}
    monkeypatch.setattr("assistant.io.api.server.serve",
                        _fake_serve_capturing(captured))

    task = await m._start_studio_daemon()
    try:
        assert "ui_bundle" in captured, (
            "_start_studio_daemon() never passed ui_bundle= to serve() -- the "
            "daemon serves no UI at all"
        )
        bundle = captured["ui_bundle"]
        assert bundle is not None
        assert bundle.read("index.html") is not None
        # From the directory, not from the vendored zip: the marker written
        # above is the one that came back.
        assert bundle.manifest()["contract"] == "does-not-have-to-match"
    finally:
        await task


@pytest.mark.asyncio
async def test_startup_falls_back_to_the_vendored_zip(tmp_path, monkeypatch):
    import assistant.config as config
    import assistant.main as m

    if not m._VENDORED_UI_ZIP.is_file():
        pytest.skip("no vendored bundle in this checkout")

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    monkeypatch.setattr(config, "STUDIO_UI_PATH", "")

    captured: dict = {}
    monkeypatch.setattr("assistant.io.api.server.serve",
                        _fake_serve_capturing(captured))

    task = await m._start_studio_daemon()
    try:
        bundle = captured.get("ui_bundle")
        assert bundle is not None, "the vendored zip was not picked up"
        assert bundle.read("index.html") is not None
    finally:
        await task


@pytest.mark.asyncio
async def test_an_unreadable_bundle_leaves_the_daemon_running_without_a_ui(
    tmp_path, monkeypatch, caplog,
):
    """The packaged-wrong case. A corrupt or absent bundle must cost the pages
    and nothing else -- the API is the product, the UI is a convenience, and a
    daemon that refuses to boot because a zip is truncated is strictly worse
    than one that boots and says so."""
    import logging

    import assistant.config as config
    import assistant.main as m

    # An export with no marker in it -- the shape a raw `next build` leaves,
    # and the one `UiBundle` refuses because it cannot tell a bundle from any
    # other directory without one.
    unreadable = tmp_path / "broken"
    unreadable.mkdir()
    (unreadable / "index.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    monkeypatch.setattr(config, "STUDIO_UI_PATH", str(unreadable))
    # And a truncated vendored zip, so there is nothing to fall back to either.
    corrupt_zip = tmp_path / "studio_ui.zip"
    corrupt_zip.write_bytes(b"not a zip at all")
    monkeypatch.setattr(m, "_VENDORED_UI_ZIP", corrupt_zip)

    captured: dict = {}
    monkeypatch.setattr("assistant.io.api.server.serve",
                        _fake_serve_capturing(captured))

    with caplog.at_level(logging.WARNING):
        task = await m._start_studio_daemon()
    try:
        assert task is not None, "the daemon must still start"
        assert captured.get("ui_bundle") is None
        assert any("no Studio UI bundle" in r.message for r in caplog.records)
    finally:
        await task


@pytest.mark.asyncio
async def test_stop_studio_daemon_clears_the_pair_store():
    """Fix 3: a stale store left set after stop is the "prints a code
    nobody can redeem" failure /studio pair exists to prevent, reached
    through a global instead of a fresh construction."""
    import assistant.main as m
    from assistant.io.api.pairing import PairCodeStore

    m._studio_pair_store = PairCodeStore()
    await m._stop_studio_daemon(None)
    assert m._studio_pair_store is None


# ─── Fix round (Task 11 review): C1 — slash commands must not execute for
# the "studio" source ───────────────────────────────────────────────────────
# POST /v1/chat requires only CHAT_SEND (routes/chat.py), and
# _StudioDispatch.submit() -> process_text_from_queue applies no further
# check per source. CHAT_SEND rides in /studio pair's own default grant
# set, so a device holding nothing else could reach the entire slash
# surface -- /studio pair, /studio revoke, /set, /reset -- with zero
# intersection against its own grants: the sideways escalation Task 10
# closed on POST /v1/pair/code, reopened through chat. Proved behaviourally
# below, not just by refusal text: the thing a slash command would have
# mutated (here, the pair store) is asserted untouched.
#
# `_telemetry.TurnTracker.save()` is stubbed out because it is a real write
# through the storage singleton -- this suite must not touch the developer's
# actual `~/TENKA/memory/tenka.db` (see this file's own docstring/rules
# above about process_text_from_queue being "a heavy, real pipeline").
# `tts.speak()` is stubbed for the same class of reason (real Kokoro
# synthesis + sounddevice playback), not because its behaviour is under
# test here.


class _FakeBridge:
    async def send_command(self, *args, **kwargs) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [
    "/studio pair evilphone",
    "/studio revoke all confirm",
    "/set followup_timer 999",
])
async def test_studio_source_refuses_slash_commands_and_executes_nothing(
    command, monkeypatch,
):
    import assistant.main as m
    from assistant.io.api.pairing import PairCodeStore

    store = PairCodeStore()
    monkeypatch.setattr(m, "_studio_pair_store", store)
    monkeypatch.setattr(m, "_wake_listener", None)
    monkeypatch.setattr(m._telemetry.TurnTracker, "save", lambda self: None)

    async def _fake_speak(text, bridge, emotion="neutral"):
        return True
    monkeypatch.setattr(m.tts, "speak", _fake_speak)

    handled: list[str] = []
    real_handle = m.slash_commands.handle

    def _tracking_handle(text):
        handled.append(text)
        return real_handle(text)
    monkeypatch.setattr(m.slash_commands, "handle", _tracking_handle)

    await m.process_text_from_queue("studio", command, _FakeBridge())

    assert handled == [], (
        f"slash_commands.handle must never run for the studio source, got: {handled}"
    )
    assert store.current() is None, "no code may be minted through the chat/API source"


@pytest.mark.asyncio
async def test_studio_source_cannot_revoke_devices_through_chat(tmp_path, monkeypatch):
    """The same property as above, proved on a real vault rather than just
    "handle() was never called": a device paired before this turn must
    still be there afterwards."""
    import assistant.main as m
    import assistant.config as config
    from assistant.io.api.vault import Capability, TokenVault

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))

    monkeypatch.setattr(m, "_wake_listener", None)
    monkeypatch.setattr(m._telemetry.TurnTracker, "save", lambda self: None)

    async def _fake_speak(text, bridge, emotion="neutral"):
        return True
    monkeypatch.setattr(m.tts, "speak", _fake_speak)

    await m.process_text_from_queue("studio", "/studio revoke all confirm", _FakeBridge())

    assert len(TokenVault(tmp_path).devices()) == 1, (
        "a studio-sourced slash command revoked a real device"
    )


@pytest.mark.asyncio
async def test_chat_source_still_executes_slash_commands(monkeypatch):
    """The mirror of the refusal above: the console/local overlay's "chat"
    source (a different thing from the Studio API's "studio" source, see
    main.py's `_input_queue.put(("chat", ...))` at the chat_message event
    handler) must be unaffected -- this is a console/voice affordance, not
    a blanket ban on slash commands."""
    import assistant.main as m
    from assistant.io.api.pairing import PairCodeStore

    store = PairCodeStore()
    monkeypatch.setattr(m, "_studio_pair_store", store)
    monkeypatch.setattr(m, "_wake_listener", None)
    monkeypatch.setattr(m._telemetry.TurnTracker, "save", lambda self: None)

    printed: list[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)),
    )

    # `grants=` is what a real "chat" item carries: `_grants_for_item` hands
    # every local source `frozenset(Capability)`. Stated here because the
    # slash branch grew a second, capability-based check behind the source
    # check in the pre-dispatch gate work -- the slash surface writes runtime
    # config, which PATCH /v1/settings charges SYSTEM_CONTROL for, so it
    # charges the same. Calling with no grants at all means "nobody said",
    # which refuses everything by design and is not a shape any producer
    # actually puts on the queue.
    from assistant.actions import LOCAL_GRANTS

    await m.process_text_from_queue("chat", "/studio pair phone", _FakeBridge(),
                                    grants=LOCAL_GRANTS)

    assert store.current() is not None, "the chat source must still be able to mint"
    assert any("Pair code minted" in line for line in printed)


# ─── Fix round: the studio-sourced refusal must reach the transcript ───────
# Task 11's refusal above was correct but silent to storage: it returned
# before memory.save_turn(), so Studio -- which settles a turn by re-reading
# the conversation transcript (LiveChatRuntime.conversation() ->
# memory.get_recent(conversation_id); POST /v1/chat is 202 Accepted with no
# turn payload) -- kept showing whatever turn was last recorded, paired
# against a message that had nothing to do with it. Asserted against the
# store itself, never the function's return value, since that return value
# was never what Studio reads.


@pytest.mark.asyncio
async def test_studio_slash_refusal_is_recorded_for_studio_to_render(tmp_path, monkeypatch):
    import assistant.config as config
    import assistant.main as m
    import assistant.memory as memory_mod
    import assistant.session as session_mod
    from assistant.storage.db import _reset_for_testing, init_db

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    _reset_for_testing()
    memory_mod._repo = None
    init_db(tmp_path / "test.db")

    monkeypatch.setattr(m, "_wake_listener", None)
    monkeypatch.setattr(m._telemetry.TurnTracker, "save", lambda self: None)

    async def _fake_speak(text, bridge, emotion="neutral"):
        return True
    monkeypatch.setattr(m.tts, "speak", _fake_speak)

    command = "/set followup_timer 999"
    await m.process_text_from_queue("studio", command, _FakeBridge())

    turns = memory_mod.get_recent(5, session_id=session_mod.get_current_session_id())
    assert turns, "the refusal must land in the transcript store, not just be spoken/returned"
    last = turns[-1]
    assert last["user_input"] == command
    assert "Slash commands aren't available" in last["response"], (
        "Studio's pane must render the refusal itself, not a stale prior turn"
    )

    _reset_for_testing()
    memory_mod._repo = None


@pytest.mark.asyncio
async def test_studio_slash_refusal_is_not_spoken_aloud(tmp_path, monkeypatch):
    """Design decision: studio is a text-native remote channel with its own
    rendered surface (the Studio pane), the same reasoning "chat" already
    gets exempted from speech for. A device holding nothing but CHAT_SEND
    can already reach this refusal on every failed attempt -- if it also
    made the local speaker narrate, that device would have a standing way
    to make TENKA talk in the owner's room on demand. The refusal is still
    fully visible where it was asked, via the transcript save proved above.
    """
    import assistant.config as config
    import assistant.main as m
    import assistant.memory as memory_mod
    from assistant.storage.db import _reset_for_testing, init_db

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    _reset_for_testing()
    memory_mod._repo = None
    init_db(tmp_path / "test.db")

    monkeypatch.setattr(m, "_wake_listener", None)
    monkeypatch.setattr(m._telemetry.TurnTracker, "save", lambda self: None)

    spoken: list[str] = []

    async def _fake_speak(text, bridge, emotion="neutral"):
        spoken.append(text)
        return True
    monkeypatch.setattr(m.tts, "speak", _fake_speak)

    await m.process_text_from_queue("studio", "/set followup_timer 999", _FakeBridge())

    assert spoken == [], f"a studio-sourced refusal must not reach tts.speak(), got: {spoken}"

    _reset_for_testing()
    memory_mod._repo = None


# ─── Fix round: the refusal must also *settle* the turn it answered ─────────
# Live-testing 6a found the two fixes above were correct and still left the
# pane hung: Studio settles a live turn on a quiet window of `status` frames
# (hooks/useEventStream.ts -- every frame re-arms a timer, a terminal phase
# starts the window, settleLiveTurn() then re-reads the transcript), and the
# only producer on a chat turn's path was `tts.speak()`, incidentally
# (SPEAKING on entry, IDLE on exit). Exempting "studio" from speech removed
# the turn's only end-of-turn signal along with it: the refusal was saved,
# `mark_done()` cleared the busy flag, and the pane sat on a placeholder with
# the composer stuck on "stop" forever.
#
# Fixed as a class, not as a branch: `_publish_turn_status()` brackets every
# "studio"-sourced turn with THINKING -> IDLE from process_text_from_queue's
# own try/finally (and mirrored in _process_one_queued_item, the same way
# mark_done() is), so every early return that answers without speaking --
# today's and tomorrow's -- settles.


@pytest.fixture()
def status_frames(monkeypatch):
    """Capture what actually reaches the event socket's producer.

    Subscribes to the real `status_broadcaster` singleton rather than
    stubbing it, because the two behaviours under test are the
    broadcaster's own: identical consecutive events are DEDUPED (which is
    why a lone terminal frame is not enough), and every frame is fanned out
    to the subscriber `main.py` hands the EventHub. `subscribe()` has no
    unsubscribe, so the list is swapped out via monkeypatch instead --
    otherwise a captor here would keep firing for the rest of the session.

    The dedupe/rate-limit state is reset too: it is module-global, and a
    frame published by an earlier test in this file would otherwise decide
    whether this one's first frame survives.
    """
    from assistant.io.status_broadcaster import status as bus

    frames: list[dict] = []
    monkeypatch.setattr(bus, "_subscribers", [frames.append])
    monkeypatch.setattr(bus, "_last_event", None)
    monkeypatch.setattr(bus, "_last_ts_per_phase", {})
    return frames


@pytest.fixture()
def studio_pipeline(tmp_path, monkeypatch):
    """A real pipeline turn against a tmp DB, with audio and telemetry off.

    Same scaffolding the refusal tests above build inline: an isolated
    SQLite file (never the developer's own ~/TENKA/memory/tenka.db), no
    wake listener, no TurnTracker write, no Kokoro/sounddevice.
    """
    import assistant.config as config
    import assistant.main as m
    import assistant.memory as memory_mod
    from assistant.storage.db import _reset_for_testing, init_db

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    _reset_for_testing()
    memory_mod._repo = None
    init_db(tmp_path / "test.db")

    monkeypatch.setattr(m, "_wake_listener", None)
    monkeypatch.setattr(m._telemetry.TurnTracker, "save", lambda self: None)

    spoken: list[str] = []

    async def _fake_speak(text, bridge, emotion="neutral"):
        spoken.append(text)
        return True
    monkeypatch.setattr(m.tts, "speak", _fake_speak)

    yield spoken

    _reset_for_testing()
    memory_mod._repo = None


TERMINAL_PHASES = {"IDLE", "DONE", "STOPPED"}  # useEventStream.ts's own set


@pytest.mark.asyncio
async def test_studio_slash_refusal_publishes_a_status_transition(
    studio_pipeline, status_frames,
):
    """The bug the user saw: no reply, composer stuck on "stop". The pane
    needs frames, and it needs the last of them to be terminal -- that is
    what arms SETTLE_QUIET_MS and eventually calls settleLiveTurn()."""
    import assistant.main as m

    await m.process_text_from_queue("studio", "/help", _FakeBridge())

    phases = [f["phase"] for f in status_frames]
    assert phases, "a studio-sourced refusal published no status frame at all"
    assert phases[-1] in TERMINAL_PHASES, (
        f"the turn never ended for a settle-on-quiet client, phases: {phases}"
    )
    assert any(p not in TERMINAL_PHASES for p in phases), (
        "the pane must see the turn begin as well as end -- a terminal-only "
        f"sequence cannot survive the broadcaster's dedupe, phases: {phases}"
    )


@pytest.mark.asyncio
async def test_a_refusal_after_a_completed_turn_still_publishes_its_end(
    studio_pipeline, status_frames,
):
    """The case a single closing frame would silently lose. Every completed
    turn leaves the bus resting at bare IDLE (tts.speak's exit, every
    da_handler's exit), and `StatusBroadcaster.set()` drops an event
    identical to the last one -- so a refusal that published only IDLE would
    publish nothing at all in the ordinary case of typing a slash command
    after a normal reply. Primed here exactly that way."""
    import assistant.main as m
    from assistant.io.status_broadcaster import StatusPhase, status as bus

    bus.set(StatusPhase.IDLE)          # a previous turn just finished
    assert status_frames, "sanity: the priming frame itself must have landed"
    status_frames.clear()

    await m.process_text_from_queue("studio", "/set studio false", _FakeBridge())

    phases = [f["phase"] for f in status_frames]
    assert phases, (
        "the refusal published nothing -- a lone terminal frame was deduped "
        "against the resting IDLE the previous turn left behind"
    )
    assert phases[-1] in TERMINAL_PHASES, f"phases: {phases}"


@pytest.mark.asyncio
async def test_the_terminal_frame_arrives_after_the_transcript_is_written(
    studio_pipeline, status_frames,
):
    """Ordering, not just presence. The terminal frame is what sends the
    client back to `GET /v1/conversations/{id}`, so the refusal has to be in
    the transcript BEFORE it fires -- otherwise the pane re-reads a
    conversation that does not contain the reply yet and settles on the
    previous turn, which is the exact stale-bubble failure the save above
    was added to fix. Asserted by reading the store from inside the
    subscriber, at the instant the frame is published."""
    import assistant.main as m
    import assistant.memory as memory_mod
    import assistant.session as session_mod
    from assistant.io.status_broadcaster import status as bus

    seen_at_publish: list[tuple[str, list]] = []

    def _capture(event: dict) -> None:
        status_frames.append(event)
        seen_at_publish.append((
            event["phase"],
            memory_mod.get_recent(5, session_id=session_mod.get_current_session_id()),
        ))
    bus._subscribers[:] = [_capture]

    command = "/studio revoke all confirm"
    await m.process_text_from_queue("studio", command, _FakeBridge())

    terminal = [turns for phase, turns in seen_at_publish if phase in TERMINAL_PHASES]
    assert terminal, "no terminal frame was published"
    turns = terminal[-1]
    assert turns, "the terminal frame fired before anything was written"
    last = turns[-1]
    assert last["user_input"] == command
    assert "Slash commands aren't available" in last["response"], (
        "the transcript a settle would re-read does not hold this turn's reply"
    )


@pytest.mark.asyncio
async def test_the_status_frames_carry_no_user_content(
    studio_pipeline, status_frames,
):
    """`detail` is RECALL-class and blanked per-socket (io/api/events.py), so
    a device paired only to watch would never see whatever is put there --
    and the refusal sentence is content, not status: it already reaches the
    pane as a transcript message. Echoing it here would render it twice for
    one device class and as an empty pill for the other."""
    import assistant.main as m

    await m.process_text_from_queue("studio", "/set followup_timer 999", _FakeBridge())

    assert status_frames, "sanity: frames must have been published"
    for frame in status_frames:
        assert frame["detail"] == "", (
            f"a turn-lifecycle frame carried a detail: {frame['detail']!r}"
        )
        assert frame["step"] is None and frame["tier"] is None


@pytest.mark.asyncio
async def test_settling_the_turn_did_not_reopen_the_slash_surface(
    studio_pipeline, status_frames, tmp_path, monkeypatch,
):
    """The security property the settle fix must not spend: still refused,
    still nothing executed, still not spoken. Proved against the real vault
    and the real pair store rather than the refusal text."""
    import assistant.main as m
    from assistant.io.api.pairing import PairCodeStore
    from assistant.io.api.vault import Capability, TokenVault

    spoken = studio_pipeline
    store = PairCodeStore()
    monkeypatch.setattr(m, "_studio_pair_store", store)
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))

    handled: list[str] = []
    real_handle = m.slash_commands.handle
    monkeypatch.setattr(
        m.slash_commands, "handle",
        lambda text: (handled.append(text), real_handle(text))[1],
    )

    await m.process_text_from_queue("studio", "/studio pair evilphone", _FakeBridge())

    assert status_frames, "sanity: the settle frames under test must have fired"
    assert handled == [], f"slash_commands.handle ran for the studio source: {handled}"
    assert store.current() is None, "a code was minted through the chat source"
    assert len(TokenVault(tmp_path).devices()) == 1
    assert spoken == [], f"the refusal was spoken aloud: {spoken}"


# ─── The siblings: every other early return that answers without speaking ──
# Same defect, same cause. The bracket covers them because it lives in the
# turn's own try/finally rather than in the branch that was reported.


@pytest.mark.asyncio
async def test_a_turn_ignored_during_recording_still_ends_for_studio(
    studio_pipeline, status_frames, monkeypatch,
):
    """The recording-mode guard is the one sibling that returns without
    speaking AND without saving: input is deliberately dropped while a
    recording session is live. Silent is fine locally; for a remote caller
    it meant the pane hung on a placeholder with no way out. It must still
    settle -- Studio drops an empty placeholder on settle, so the user gets
    an unanswered message rather than a permanent "stop" button."""
    import assistant.main as m
    from assistant.intent import IntentResult

    monkeypatch.setattr(m.recording, "is_active", lambda: True)

    # The guard sits downstream of intent classification, so the classifier
    # and the spaCy-backed topic tracker are stubbed out: neither is under
    # test here, and this suite makes no network calls and loads no models.
    async def _fake_intent(*args, **kwargs):
        return IntentResult(intent="get_time", response="", params={})
    monkeypatch.setattr(m, "detect_intent", _fake_intent)

    class _NoTopics:
        def resolve_query(self, text):
            return text

        def get_topic_hint(self):
            return ""

        def push_turn(self, text, turn_number):
            return None
    monkeypatch.setattr(m, "_get_topic_tracker", lambda: _NoTopics())

    await m.process_text_from_queue("studio", "what is the time", _FakeBridge())

    phases = [f["phase"] for f in status_frames]
    assert phases and phases[-1] in TERMINAL_PHASES, (
        f"a studio turn dropped by the recording guard never ended: {phases}"
    )
    assert studio_pipeline == [], "the recording guard must stay silent"


@pytest.mark.asyncio
async def test_a_raise_before_the_pipelines_own_try_still_ends_the_turn(
    status_frames, monkeypatch,
):
    """The mirror of the busy-flag test above. Three things run before
    process_text_from_queue's own `try:` (TurnTracker construction,
    set_current_tracker, pausing the wake listener); a raise in any of them
    skips that function's `finally` entirely. `_process_one_queued_item`
    already exists to clear the busy flag on that path -- the turn's end has
    to reach the pane the same way, or the composer unblocks against a
    bubble that never resolves."""
    import assistant.main as m

    def _boom(*args, **kwargs):
        raise RuntimeError("tracker construction failed")
    monkeypatch.setattr(m._telemetry, "TurnTracker", _boom)

    with pytest.raises(RuntimeError):
        await m._process_one_queued_item(("studio", "hello"), _FakeBridge())

    phases = [f["phase"] for f in status_frames]
    assert phases and phases[-1] in TERMINAL_PHASES, (
        f"a raise before the inner try left the turn open forever: {phases}"
    )


@pytest.mark.asyncio
async def test_local_sources_gain_no_frames_from_the_bracket(
    studio_pipeline, status_frames,
):
    """Scoped to the channel that has a settle model. The desktop overlay
    reads the same bus and renders whatever pill the running handler
    publishes; a second producer on every voice turn would only risk a
    spurious "Done" flash on transitions that did no work. Driven through
    the "chat" source's slash path, which executes and prints rather than
    refusing."""
    import assistant.main as m

    await m.process_text_from_queue("chat", "/set followup_timer 7", _FakeBridge())

    assert status_frames == [], (
        f"the bracket must not fire for a local source, got: {status_frames}"
    )
