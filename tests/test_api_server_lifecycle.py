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
    assert "/v1/memory/{scope}" in schema["paths"]


# ─── _StudioDispatch — the only path from a request into the pipeline ─────
# Importing assistant.main only loads its module (functions, classes,
# logging setup) -- it never starts the assistant. Nothing here calls
# main() or async_main(), so no microphone, no desktop control, no API
# spend.


@pytest.mark.asyncio
async def test_studio_dispatch_puts_the_shape_the_consumer_loop_expects():
    """main.py's queue consumer (~line 1953) does
    `source, text = item[0], item[1]`, the same 2-tuple shape used for
    the existing "chat" source -- not the brief's guessed 3-tuple with a
    trailing None."""
    import assistant.main as m

    while True:
        try:
            m._input_queue.get_nowait()
        except Exception:
            break

    dispatch = m._StudioDispatch()
    turn_id, session_id, accepted, reason = await dispatch.submit("hello studio")
    assert (accepted, reason) == (True, "")
    assert turn_id == "studio-1"
    assert isinstance(session_id, str)  # get_current_session_id()'s real return type

    item = m._input_queue.get_nowait()
    assert item == ("studio", "hello studio")


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
        dispatch.submit("first"), dispatch.submit("second"),
    )
    accepted = [r for r in results if r[2] is True]
    refused = [r for r in results if r[2] is False]
    assert len(accepted) == 1, f"expected exactly one acceptance, got {results}"
    assert len(refused) == 1, f"expected exactly one refusal, got {results}"
    assert refused[0] == ("", "", False, "busy")

    # Still busy: the accepted turn has not finished (mark_done() has not
    # run) -- a third submit must still be refused.
    still_busy = await dispatch.submit("third")
    assert still_busy[2] is False

    dispatch.mark_done()
    now_free = await dispatch.submit("fourth")
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
    finally_block = source[finally_idx: finally_idx + 800]
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

    turn_id, _, accepted, _ = await dispatch.submit("first")
    assert accepted is True
    assert dispatch.busy is True

    with pytest.raises(RuntimeError):
        await m._process_one_queued_item(("studio", "first"), bridge=None)

    assert dispatch.busy is False, (
        "a raise before process_text_from_queue's own try: left the busy flag stranded"
    )

    second = await dispatch.submit("second")
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

    async def _fake_process(source, text, bridge, stt_ms=None):
        return None

    monkeypatch.setattr(m, "process_text_from_queue", _fake_process)

    await dispatch.submit("hello")
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

    async def _slow_process(source, text, bridge, stt_ms=None):
        await release.wait()

    monkeypatch.setattr(m, "process_text_from_queue", _slow_process)

    await dispatch.submit("hello")
    task = asyncio.create_task(m._process_one_queued_item(("studio", "hello"), bridge=None))
    await asyncio.sleep(0)  # let the task actually start awaiting release

    still_busy = await dispatch.submit("second")
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
    process, on a machine where the daemon is off."""
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
