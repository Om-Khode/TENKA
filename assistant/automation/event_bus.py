"""Event-driven monitor bus.

Runs a single daemon thread with a Win32 message pump. Receives OS events
(SMTC media changes, WinEventHook window focus/title) and evaluates
user-defined monitors against them. Actions fire via proactive queue (TTS)
or actions.execute (code_executor) on the main async loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger("event_bus")


# ─── Condition Evaluation ────────────────────────────────────────────────────

def compile_condition(expr: str) -> Any | None:
    try:
        return compile(expr, "<monitor>", "eval")
    except SyntaxError:
        return None


_SAFE_BUILTINS = {
    "any": any, "all": all, "len": len, "ord": ord, "chr": chr,
    "min": min, "max": max, "abs": abs, "sum": sum, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "isinstance": isinstance, "range": range, "sorted": sorted,
    "True": True, "False": False, "None": None,
}


class _EventNS:
    """Attribute-accessible wrapper so both `title` and `event.title` work."""
    def __init__(self, d: dict):
        self.__dict__.update(d)


def eval_condition_code(compiled: Any, event_locals: dict) -> bool:
    try:
        ns = {**event_locals, "event": _EventNS(event_locals)}
        return bool(eval(compiled, {"__builtins__": _SAFE_BUILTINS}, ns))
    except Exception:
        return False


# ─── Dispatch Helpers ────────────────────────────────────────────────────────

def make_dedup_key(event: dict) -> str:
    etype = event.get("event_type", "")
    if etype == "media_changed":
        return f"{event.get('title', '')}|{event.get('artist', '')}"
    if etype in ("window_focus", "window_title"):
        return f"{event.get('source_app', '')}|{event.get('window_title', '')}"
    return ""


def check_dispatch(monitor: dict, event: dict, *, now: float) -> bool:
    if monitor["event_type"] != event.get("event_type"):
        return False

    src_filter = monitor.get("source_filter")
    if src_filter and src_filter.lower() not in event.get("source_app", "").lower():
        return False

    dedup_key = make_dedup_key(event)
    if dedup_key == monitor.get("_last_dedup_key"):
        last_fire = monitor.get("_last_fire_time", 0)
        if now - last_fire < monitor.get("cooldown_secs", 5):
            return False

    return True


def render_payload(template: str, event: dict) -> str:
    try:
        return template.format(**event)
    except (KeyError, IndexError):
        return template


# ─── EventBus Class ──────────────────────────────────────────────────────────

class EventBus:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._thread_id: int = 0
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._active_monitors: list[dict] = []
        self._compiled_conditions: dict[int, Any] = {}
        self._stop_requested = False
        self._debounce_timers: dict[int, threading.Timer] = {}
        self._debounce_events: dict[int, tuple[dict, dict]] = {}

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        from assistant.storage.db import get_db
        from assistant.storage.repos.monitor import MonitorRepo

        self._main_loop = loop
        db = get_db()
        if db is None:
            logger.warning("[event-monitor] DB not initialized, skipping EventBus start")
            return

        self._load_monitors(MonitorRepo(db))

        self._thread = threading.Thread(
            target=self._run_message_pump,
            name="em1-event-bus",
            daemon=True,
        )
        self._thread.start()
        logger.info("[event-monitor] EventBus started (%d active monitors)", len(self._active_monitors))

    def stop(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop_requested = True
        for timer in self._debounce_timers.values():
            timer.cancel()
        self._debounce_timers.clear()
        self._debounce_events.clear()
        import ctypes
        WM_QUIT = 0x0012
        ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        self._thread.join(timeout=5)
        logger.info("[event-monitor] EventBus stopped")

    def reload_monitors(self, *, flush_pending: bool = False) -> None:
        from assistant.storage.db import get_db
        from assistant.storage.repos.monitor import MonitorRepo

        if flush_pending:
            self._flush_pending_tts()
        db = get_db()
        if db is None:
            return
        self._load_monitors(MonitorRepo(db))
        logger.info("[event-monitor] Reloaded monitors (%d active)", len(self._active_monitors))

    def _flush_pending_tts(self) -> None:
        from assistant import proactive
        flushed = 0
        while not proactive._proactive_queue.empty():
            try:
                proactive._proactive_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        if flushed:
            logger.info("[event-monitor] Flushed %d pending notifications", flushed)

    def _load_monitors(self, repo) -> None:
        monitors = repo.get_active()
        for m in monitors:
            expr = m.get("condition_expr")
            if expr and m.get("condition_mode") == "code":
                compiled = compile_condition(expr)
                if compiled is not None:
                    self._compiled_conditions[m["id"]] = compiled
                    logger.info("[event-monitor] Compiled condition for #%d '%s': %s",
                                m["id"], m.get("name", "?"), expr)
                else:
                    logger.warning("[event-monitor] FAILED to compile condition for #%d '%s': %s",
                                   m["id"], m.get("name", "?"), expr)
        self._active_monitors = monitors

    def _dispatch_event(self, event: dict) -> None:
        etype = event.get("event_type", "?")
        logger.debug("[event-monitor] Event received: %s | %s", etype, make_dedup_key(event))
        now = time.time()
        for monitor in self._active_monitors:
            if not check_dispatch(monitor, event, now=now):
                continue

            matched = self._eval_condition(monitor, event)
            if not matched:
                logger.debug("[event-monitor] Condition=False for #%d '%s'",
                             monitor["id"], monitor.get("name", "?"))
                continue

            monitor["_last_fire_time"] = now
            monitor["_last_dedup_key"] = make_dedup_key(event)

            is_media = etype == "media_changed"
            is_tts = monitor.get("action_type") == "tts_notify"
            if is_media and is_tts:
                self._debounce_fire(monitor, event)
            else:
                self._fire_action(monitor, event)

    def _debounce_fire(self, monitor: dict, event: dict) -> None:
        from assistant import config
        mid = monitor["id"]
        existing = self._debounce_timers.pop(mid, None)
        if existing is not None:
            existing.cancel()
            logger.info("[event-monitor] Debounce reset for monitor #%d", mid)
        self._debounce_events[mid] = (monitor, event)
        delay = getattr(config, "EVENT_MONITOR_DEBOUNCE_SECS", 2.0)
        timer = threading.Timer(delay, self._debounce_callback, args=[mid])
        timer.daemon = True
        timer.start()
        self._debounce_timers[mid] = timer

    def _debounce_callback(self, monitor_id: int) -> None:
        pair = self._debounce_events.pop(monitor_id, None)
        self._debounce_timers.pop(monitor_id, None)
        if pair is not None:
            monitor, event = pair
            monitor["_last_fire_time"] = time.time()
            self._fire_action(monitor, event)

    def _eval_condition(self, monitor: dict, event: dict) -> bool:
        mid = monitor["id"]
        mode = monitor.get("condition_mode", "code")

        if mode == "code":
            compiled = self._compiled_conditions.get(mid)
            if compiled is not None:
                return eval_condition_code(compiled, event)
            if not monitor.get("condition_expr"):
                prompt = monitor.get("condition_prompt")
                if prompt:
                    return self._eval_condition_llm_sync(prompt, event)
                return True
            prompt = monitor.get("condition_prompt")
            if prompt:
                return self._eval_condition_llm_sync(prompt, event)
            return False

        if mode == "llm":
            prompt = monitor.get("condition_prompt")
            if prompt:
                return self._eval_condition_llm_sync(prompt, event)
        return False

    def _eval_condition_llm_sync(self, prompt: str, event: dict) -> bool:
        if self._main_loop is None:
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._eval_condition_llm(prompt, event),
                self._main_loop,
            )
            return future.result(timeout=15)
        except Exception as e:
            logger.info("[event-monitor] LLM condition eval failed: %s", e)
            return False

    async def _eval_condition_llm(self, prompt: str, event: dict) -> bool:
        from assistant.llm.contracts import ask_for_synthesis

        llm_prompt = (
            f"Event data:\n{json.dumps(event, indent=2)}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer ONLY \"yes\" or \"no\"."
        )
        response = await ask_for_synthesis(llm_prompt, max_tokens=8, temperature=0.0)
        return response.strip().lower().startswith("yes")

    def _fire_action(self, monitor: dict, event: dict) -> None:
        action_type = monitor.get("action_type", "tts_notify")
        payload = render_payload(monitor.get("action_payload", ""), event)

        self._record_fire_async(monitor["id"])

        if action_type == "tts_notify":
            from assistant import proactive
            proactive._proactive_queue.put((payload, "neutral"))
            logger.info("[event-monitor] TTS notify: %s", payload[:80])
        elif action_type == "code_executor":
            if self._main_loop is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self._run_code_executor(payload),
                    self._main_loop,
                )
                future.add_done_callback(self._on_action_complete)
                logger.info("[event-monitor] Code executor fired: %s", payload[:80])

    async def _run_code_executor(self, goal: str) -> str:
        from assistant.actions import execute
        return await execute("code_executor", {"goal": goal}, "")

    def _on_action_complete(self, future: asyncio.Future) -> None:
        try:
            result = future.result(timeout=0)
            logger.info("[event-monitor] Action completed: %s", str(result)[:120])
        except Exception as e:
            logger.warning("[event-monitor] Action failed: %s", e)

    def _record_fire_async(self, monitor_id: int) -> None:
        try:
            from assistant.storage.db import get_db
            from assistant.storage.repos.monitor import MonitorRepo
            db = get_db()
            if db:
                MonitorRepo(db).record_fire(monitor_id, datetime.now().isoformat())
        except Exception as e:
            logger.debug("[event-monitor] Failed to record fire: %s", e)

    def _run_message_pump(self) -> None:
        import ctypes
        import ctypes.wintypes

        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        logger.info("[event-monitor] Pump thread started (tid=%d)", self._thread_id)

        # Register window event source
        from .event_sources.window import WindowEventSource
        self._window_source = WindowEventSource(self._dispatch_event)
        self._window_source.register()

        # Register media event source (if available)
        self._media_source = None
        try:
            from .event_sources.media import MediaEventSource
            self._media_source = MediaEventSource(
                dispatch_fn=self._dispatch_event,
                thread_id=self._thread_id,
            )
            self._media_source.start()
        except Exception as e:
            logger.warning("[event-monitor] Media source unavailable: %s", e)

        # Win32 message pump — blocks until WM_QUIT
        WM_APP_MEDIA = 0x8001  # WM_APP + 1
        msg = ctypes.wintypes.MSG()
        while ctypes.windll.user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) > 0:
            if msg.message == WM_APP_MEDIA and self._media_source is not None:
                self._media_source.handle_queued_event()
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))

        # Cleanup
        self._window_source.unregister()
        if self._media_source is not None:
            self._media_source.stop()
        logger.info("[event-monitor] Pump thread exited")


# ─── Module-Level Singleton ──────────────────────────────────────────────────

event_bus = EventBus()
