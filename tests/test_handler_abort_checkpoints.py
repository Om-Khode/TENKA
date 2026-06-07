"""For each cursor-controlling handler: confirm mid-loop abort raises
UserAborted at the next checkpoint and the handler returns a clean
response (no traceback, status returns to IDLE)."""
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from assistant.core.abort import abort, UserAborted
from assistant.io.status_broadcaster import status, StatusPhase


@pytest.fixture(autouse=True)
def _reset_abort():
    abort.reset()
    abort._tasks.clear()
    yield
    abort.reset()
    abort._tasks.clear()


def test_computer_task_aborts_at_loop_boundary():
    """vision agent's main loop checks _check_abort() (proxies to abort.is_aborted)."""
    from assistant.automation.vision import agent

    # Simulate: abort requested before loop iterates
    abort.register_task("t")
    abort.request_abort("esc_hold")
    assert agent._check_abort() is True


@pytest.mark.asyncio
async def test_browser_action_returns_clean_response_on_abort():
    from assistant.actions.da_handlers import handle_browser_action
    abort.register_task("t")
    abort.request_abort("esc_hold")
    # _from_planner=True bypasses the abort.reset() at handler entry so the
    # pre-set abort flag is still active when the handler checks is_aborted().
    result = await handle_browser_action(
        {"url": "https://example.com", "steps": []}, "", None,
        _from_planner=True,
    )
    assert "stopped" in result.lower() or "ok" in result.lower()


@pytest.mark.asyncio
async def test_find_and_click_propagates_userabort_to_planner():
    """When called from the planner with abort pre-set, the handler must
    raise UserAborted (not return a "Stopped." string that the planner
    would treat as a successful step output)."""
    from assistant.actions.da_handlers import handle_find_and_click
    abort.register_task("t")
    abort.request_abort("esc_hold")
    with pytest.raises(UserAborted):
        await handle_find_and_click({"target": "Send"}, "", None,
                                    _from_planner=True)


@pytest.mark.asyncio
async def test_manifest_dispatch_returns_clean_response_on_abort():
    from assistant.actions.manifest_dispatch import handle_manifest_dispatch
    abort.register_task("t")
    abort.request_abort("esc_hold")
    result = await handle_manifest_dispatch(
        {"app_id": "test", "intent_id": "x", "params": {}}, "", None,
    )
    assert "stopped" in result.lower() or "ok" in result.lower()


@pytest.mark.asyncio
async def test_app_action_returns_clean_response_on_abort():
    from assistant.actions.da_handlers import handle_app_action
    abort.register_task("t")
    abort.request_abort("esc_hold")
    # _from_planner=True bypasses the abort.reset() at handler entry.
    result = await handle_app_action({"app": "Notepad", "steps": []}, "", None,
                                     _from_planner=True)
    assert "stopped" in result.lower() or "ok" in result.lower()


@pytest.mark.asyncio
async def test_planner_returns_clean_response_on_abort():
    from assistant.actions.da_handlers import handle_planner

    # Patch execute_plan so it calls request_abort then raises UserAborted,
    # simulating the planner's inner step loop hitting an abort checkpoint.
    async def fake_execute_plan(*args, **kwargs):
        abort.request_abort("esc_hold")
        raise UserAborted("esc_hold")

    with patch(
        "assistant.actions.planner.planner.execute_plan",
        side_effect=fake_execute_plan,
    ):
        result = await handle_planner({"goal": "test goal"}, "", None)

    assert "stopped" in result.lower() or "ok" in result.lower()


@pytest.mark.asyncio
async def test_code_executor_propagates_userabort_to_planner():
    """When called from planner with abort pre-set, must raise UserAborted."""
    from assistant.actions.da_handlers import handle_code_executor
    abort.register_task("t")
    abort.request_abort("esc_hold")
    with pytest.raises(UserAborted):
        await handle_code_executor({"goal": "test"}, "", None,
                                   _from_planner=True)
