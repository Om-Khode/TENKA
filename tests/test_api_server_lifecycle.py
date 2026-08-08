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
    """The single-turn lock: a second submit while the first is still being
    handled must be refused outright, not queued behind it."""
    import asyncio

    import assistant.main as m

    dispatch = m._StudioDispatch()

    async def _hold_the_lock():
        async with dispatch._lock:
            await asyncio.sleep(0.2)

    holder = asyncio.create_task(_hold_the_lock())
    await asyncio.sleep(0.02)  # let the holder actually acquire first
    turn_id, session_id, accepted, reason = await dispatch.submit("should be refused")
    assert (turn_id, session_id, accepted, reason) == ("", "", False, "busy")
    await holder


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
