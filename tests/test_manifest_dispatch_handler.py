"""Smoke tests for the manifest_dispatch action handler.

Covers the two key code paths:
  1. dispatcher returns ok=True → handler returns "" (caller renders TTS)
  2. dispatcher returns ok=False → handler escalates to computer_task handle_computer_task

All app/process names are generic (`test_app.desktop`, `TestApp.exe`,
`TestApp`) per THE rule — no brand names in production code or tests.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.actions.manifest_dispatch import handle_manifest_dispatch


@pytest.mark.asyncio
async def test_handler_returns_empty_on_success():
    fake_disp = MagicMock()
    fake_disp.dispatch.return_value = MagicMock(
        ok=True, selector_used_index=0, error=None,
    )
    with patch(
        "assistant.automation.manifest_runtime.get_dispatcher",
        return_value=fake_disp,
    ), patch(
        "assistant.automation.manifest_registry.get_singleton",
        return_value=MagicMock(),
    ), patch(
        "assistant.automation.router.detect_active_app",
        return_value={
            "process_names": ["TestApp.exe"],
            "window_title": "TestApp",
            "active_url": "",
        },
    ):
        result = await handle_manifest_dispatch(
            params={
                "app_id": "test_app.desktop",
                "intent_id": "play",
                "slots": {},
            },
            llm_response="",
        )

    assert result == ""
    fake_disp.dispatch.assert_called_once()
    call_kwargs = fake_disp.dispatch.call_args.kwargs
    assert call_kwargs["app_id"] == "test_app.desktop"
    assert call_kwargs["intent_id"] == "play"
    assert call_kwargs["active_window"] == "TestApp"


@pytest.mark.asyncio
async def test_handler_escalates_to_computer_task_on_failure():
    fake_disp = MagicMock()
    fake_disp.dispatch.return_value = MagicMock(
        ok=False,
        escalate_to_dispatch=True,
        error="all selectors exhausted",
        selector_used_index=None,
    )
    fake_computer_task = AsyncMock(return_value="computer_task took over")
    with patch(
        "assistant.automation.manifest_runtime.get_dispatcher",
        return_value=fake_disp,
    ), patch(
        "assistant.automation.manifest_registry.get_singleton",
        return_value=MagicMock(),
    ), patch(
        "assistant.automation.router.detect_active_app",
        return_value={
            "process_names": ["TestApp.exe"],
            "window_title": "TestApp",
            "active_url": "",
        },
    ), patch(
        "assistant.actions.da_handlers.handle_computer_task", fake_computer_task,
    ):
        result = await handle_manifest_dispatch(
            params={
                "app_id": "test_app.desktop",
                "intent_id": "play",
                "slots": {},
            },
            llm_response="",
        )

    assert "computer_task" in result
    # Verify suffix stripping + underscore-to-space conversion
    fake_computer_task.assert_awaited_once()
    computer_task_kwargs = fake_computer_task.await_args.kwargs
    assert computer_task_kwargs["params"]["goal"] == "play in test app"


@pytest.mark.asyncio
async def test_handler_escalates_when_registry_is_none():
    fake_computer_task = AsyncMock(return_value="computer_task took over (no registry)")
    with patch("assistant.automation.manifest_registry.get_singleton",
               return_value=None):
        with patch("assistant.automation.manifest_runtime.get_dispatcher",
                   return_value=MagicMock()):
            with patch("assistant.actions.da_handlers.handle_computer_task",
                       fake_computer_task):
                result = await handle_manifest_dispatch(
                    params={"app_id": "test_app.desktop",
                            "intent_id": "play", "slots": {}},
                    llm_response="",
                )
                assert "computer_task" in result
                fake_computer_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_handler_escalates_when_dispatcher_is_none():
    fake_computer_task = AsyncMock(return_value="computer_task took over (no dispatcher)")
    with patch("assistant.automation.manifest_registry.get_singleton",
               return_value=MagicMock()):
        with patch("assistant.automation.manifest_runtime.get_dispatcher",
                   return_value=None):
            with patch("assistant.actions.da_handlers.handle_computer_task",
                       fake_computer_task):
                result = await handle_manifest_dispatch(
                    params={"app_id": "test_app.desktop",
                            "intent_id": "play", "slots": {}},
                    llm_response="",
                )
                assert "computer_task" in result
                fake_computer_task.assert_awaited_once()
