"""Tests for /promote slash command + 50-save auto-promotion counter."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from assistant import slash_commands
from assistant.automation import manifest_registry, promoter as promoter_mod, step_cache


# ─── Async test helper ────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


# ─── Shared reset fixture ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset singleton + save counter + in-flight flag between tests."""
    monkeypatch.setattr(manifest_registry, "_singleton", None)
    monkeypatch.setattr(step_cache, "_save_counter", 0)
    monkeypatch.setattr(promoter_mod, "_in_flight", False)
    yield


# ─── /promote slash command ───────────────────────────────────────────────

def test_promote_returns_uninit_message_when_registry_missing(monkeypatch):
    """Without an initialized registry, /promote bails with a clear message."""
    # Force the db lookup to also be None — even if a previous test left
    # one behind, the registry None alone should short-circuit.
    monkeypatch.setattr(
        "assistant.storage.db.get_db", lambda: None, raising=False,
    )

    result = slash_commands.handle("/promote")
    assert result == "manifest-based not initialized."


def test_promote_returns_uninit_message_when_db_missing(monkeypatch):
    """Registry set but DB None → still bails."""
    monkeypatch.setattr(manifest_registry, "_singleton", MagicMock())
    monkeypatch.setattr(
        "assistant.storage.db.get_db", lambda: None, raising=False,
    )

    result = slash_commands.handle("/promote")
    assert result == "manifest-based not initialized."


def test_promote_returns_uninit_when_no_running_loop(monkeypatch):
    """Sync caller with no running loop returns the distinct no-loop string.

    Registry/DB-missing and no-loop are different failure modes; surface
    them with different messages so callers can tell them apart.
    """
    monkeypatch.setattr(manifest_registry, "_singleton", MagicMock())
    monkeypatch.setattr(
        "assistant.storage.db.get_db", lambda: MagicMock(), raising=False,
    )

    # Called from a sync context — no running loop.
    result = slash_commands.handle("/promote")
    assert result == "Cannot schedule: no async loop available."


def test_promote_returns_busy_when_already_running(monkeypatch):
    """A second /promote while one is in flight returns the busy message
    and does NOT schedule a duplicate cycle."""
    fake_registry = MagicMock()
    fake_registry.store = MagicMock()
    monkeypatch.setattr(manifest_registry, "_singleton", fake_registry)
    monkeypatch.setattr(
        "assistant.storage.db.get_db", lambda: MagicMock(), raising=False,
    )

    # Simulate an in-progress cycle already holding the flag.
    monkeypatch.setattr(promoter_mod, "_in_flight", True)

    schedule_count = {"n": 0}

    async def _runner():
        loop = asyncio.get_running_loop()
        real_create_task = loop.create_task

        def _spy_create_task(coro, **kw):
            schedule_count["n"] += 1
            # Close the coroutine to avoid "never awaited" warnings even
            # though we don't expect to reach this branch.
            coro.close()
            f = loop.create_future()
            f.set_result(None)
            return f

        loop.create_task = _spy_create_task  # type: ignore[method-assign]
        try:
            return slash_commands.handle("/promote")
        finally:
            loop.create_task = real_create_task  # type: ignore[method-assign]

    result = _run(_runner())
    assert result == "manifest-based promotion already in progress."
    assert schedule_count["n"] == 0


def test_promote_schedules_run_once_when_loop_running(monkeypatch):
    """When called inside an async function, schedules run_once on the loop
    and returns the acknowledgment immediately."""
    fake_registry = MagicMock()
    fake_registry.store = MagicMock()
    monkeypatch.setattr(manifest_registry, "_singleton", fake_registry)
    monkeypatch.setattr(
        "assistant.storage.db.get_db", lambda: MagicMock(), raising=False,
    )

    # Replace Promoter with a stub whose run_once returns a known summary.
    fake_summary = {"apps_processed": 0, "intents_promoted": 0}

    async def _fake_run_once(self):
        return fake_summary

    monkeypatch.setattr(
        "assistant.automation.promoter.Promoter.run_once",
        _fake_run_once,
    )

    captured: dict = {"task": None}

    async def _runner():
        loop = asyncio.get_running_loop()
        real_create_task = loop.create_task

        def _spy_create_task(coro, **kw):
            t = real_create_task(coro, **kw)
            captured["task"] = t
            return t

        loop.create_task = _spy_create_task  # type: ignore[method-assign]
        try:
            result = slash_commands.handle("/promote")
        finally:
            loop.create_task = real_create_task  # type: ignore[method-assign]
        # Let the scheduled task drain.
        if captured["task"] is not None:
            await captured["task"]
        return result

    result = _run(_runner())
    assert result == "manifest-based promotion cycle scheduled. Results will be logged."
    assert captured["task"] is not None


def test_promote_returns_ack_message(monkeypatch):
    """Synchronous return value is exactly the acknowledgment string."""
    fake_registry = MagicMock()
    fake_registry.store = MagicMock()
    monkeypatch.setattr(manifest_registry, "_singleton", fake_registry)
    monkeypatch.setattr(
        "assistant.storage.db.get_db", lambda: MagicMock(), raising=False,
    )

    async def _fake_run_once(self):
        return {}

    monkeypatch.setattr(
        "assistant.automation.promoter.Promoter.run_once",
        _fake_run_once,
    )

    async def _runner():
        result = slash_commands.handle("/promote")
        # Drain any pending tasks so the event loop closes cleanly.
        await asyncio.sleep(0)
        return result

    result = _run(_runner())
    assert result == "manifest-based promotion cycle scheduled. Results will be logged."


# ─── 50-save auto-promotion counter ───────────────────────────────────────


class _FakeRepoNoop:
    def save(self, *a, **kw):
        pass


def test_save_counter_increments(monkeypatch):
    """Calling save_cached_steps 50 times triggers a single scheduled task."""
    # Fake repo that records save calls.
    save_calls: list = []

    class _FakeRepo:
        def save(self, *a, **kw):
            save_calls.append((a, kw))

    monkeypatch.setattr(step_cache, "_get_repo", lambda: _FakeRepo())

    # Registry + DB stubs so the auto-schedule path can build a Promoter.
    fake_registry = MagicMock()
    fake_registry.store = MagicMock()
    monkeypatch.setattr(manifest_registry, "_singleton", fake_registry)
    monkeypatch.setattr(
        "assistant.storage.db.get_db", lambda: MagicMock(), raising=False,
    )

    schedule_count = {"n": 0}

    async def _fake_run_once(self):
        return {"apps_processed": 0}

    monkeypatch.setattr(
        "assistant.automation.promoter.Promoter.run_once", _fake_run_once,
    )

    async def _runner():
        loop = asyncio.get_running_loop()
        real_create_task = loop.create_task
        tasks: list = []

        def _spy_create_task(coro, **kw):
            schedule_count["n"] += 1
            t = real_create_task(coro, **kw)
            tasks.append(t)
            return t

        loop.create_task = _spy_create_task  # type: ignore[method-assign]
        try:
            for i in range(50):
                step_cache.save_cached_steps(
                    "terminator", "generic_app", f"goal {i}",
                    [{"action": "press_key", "params": {"key": "Space"}}],
                )
        finally:
            loop.create_task = real_create_task  # type: ignore[method-assign]
        for t in tasks:
            await t

    _run(_runner())
    assert len(save_calls) == 50
    assert schedule_count["n"] == 1


def test_save_counter_no_loop_no_crash(monkeypatch):
    """Saving outside a running loop must not crash."""
    save_calls: list = []

    class _FakeRepo:
        def save(self, *a, **kw):
            save_calls.append((a, kw))

    monkeypatch.setattr(step_cache, "_get_repo", lambda: _FakeRepo())

    # Bump counter to one short of the threshold so the next save crosses it
    # — the no-loop branch is the one we want to exercise.
    monkeypatch.setattr(step_cache, "_save_counter", 49)

    # No event loop running here — sync test body.
    step_cache.save_cached_steps(
        "terminator", "generic_app", "do a thing",
        [{"action": "press_key", "params": {"key": "Space"}}],
    )
    assert len(save_calls) == 1


def test_save_counter_under_threshold_no_schedule(monkeypatch):
    """Under N saves → no scheduling attempt, even with a running loop."""
    monkeypatch.setattr(step_cache, "_get_repo", lambda: _FakeRepoNoop())

    schedule_count = {"n": 0}

    async def _runner():
        loop = asyncio.get_running_loop()
        real_create_task = loop.create_task

        def _spy_create_task(coro, **kw):
            schedule_count["n"] += 1
            return real_create_task(coro, **kw)

        loop.create_task = _spy_create_task  # type: ignore[method-assign]
        try:
            for i in range(49):
                step_cache.save_cached_steps(
                    "terminator", "generic_app", f"goal {i}", [],
                )
        finally:
            loop.create_task = real_create_task  # type: ignore[method-assign]

    _run(_runner())
    assert schedule_count["n"] == 0


# ─── Registry public accessors ────────────────────────────────────────────


def test_registry_exposes_store_and_index_repo():
    """ManifestRegistry.store / .index_repo return the constructor args."""
    fake_store = MagicMock()
    fake_repo = MagicMock()
    registry = manifest_registry.ManifestRegistry(
        store=fake_store, index_repo=fake_repo,
    )
    assert registry.store is fake_store
    assert registry.index_repo is fake_repo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
