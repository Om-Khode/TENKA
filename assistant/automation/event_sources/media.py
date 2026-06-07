"""SMTC media event source for.

Uses Windows.Media.Control (WinRT) to detect media/track changes.
Callbacks fire on a WinRT threadpool thread and post WM_APP+1 to
the EventBus pump thread via PostThreadMessageW. The pump calls
handle_queued_event() on its own thread to dispatch safely.
"""
from __future__ import annotations

import ctypes
import logging
from datetime import datetime
from typing import Callable

logger = logging.getLogger("event_bus.media")

WM_APP_MEDIA = 0x8001  # WM_APP + 1


def _map_playback_status(status_value: int) -> str:
    STATUS_MAP = {0: "closed", 1: "opened", 2: "changing", 3: "stopped", 4: "playing", 5: "paused"}
    return STATUS_MAP.get(status_value, "unknown")


class MediaEventSource:
    name: str = "smtc"
    event_types: frozenset[str] = frozenset({"media_changed"})

    def __init__(self, dispatch_fn: Callable[[dict], None], thread_id: int) -> None:
        self._dispatch = dispatch_fn
        self._thread_id = thread_id
        self._manager = None
        self._current_session = None
        self._pending_event: dict | None = None

    def start(self, dispatch_fn: Callable[[dict], None] | None = None, **kwargs) -> None:
        import asyncio
        from winsdk.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as SessionManager,
        )

        loop = asyncio.new_event_loop()
        self._manager = loop.run_until_complete(
            SessionManager.request_async()
        )
        loop.close()

        self._manager.add_current_session_changed(self._on_session_changed)
        self._attach_current_session()
        logger.info("[event-monitor] SMTC media source started")

    def stop(self) -> None:
        self._current_session = None
        self._manager = None
        logger.info("[event-monitor] SMTC media source stopped")

    def _attach_current_session(self) -> None:
        if self._manager is None:
            return
        session = self._manager.get_current_session()
        if session is None:
            self._current_session = None
            return
        self._current_session = session
        session.add_media_properties_changed(self._on_media_changed)
        # DEBUG (was INFO) — SMTC fires this on every active-session swap,
        # which can be many times a minute when the user alt-tabs between
        # media apps. Keep diagnostic-reachable via logging level, off the
        # default INFO console.
        logger.debug("[event-monitor] Attached to media session: %s",
                     session.source_app_user_model_id or "unknown")

    def _on_session_changed(self, sender, args) -> None:
        self._attach_current_session()

    def _on_media_changed(self, session, args) -> None:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            props = loop.run_until_complete(
                session.try_get_media_properties_async()
            )
            loop.close()

            playback_info = session.get_playback_info()
            status_val = playback_info.playback_status if playback_info else 0

            self._pending_event = {
                "event_type": "media_changed",
                "source_app": session.source_app_user_model_id or "",
                "title": props.title or "",
                "artist": props.artist or "",
                "album": props.album_title or "",
                "playback_status": _map_playback_status(status_val),
                "timestamp": datetime.now().isoformat(),
            }

            ctypes.windll.user32.PostThreadMessageW(
                self._thread_id, WM_APP_MEDIA, 0, 0,
            )
        except Exception as e:
            logger.debug("[event-monitor] SMTC callback error: %s", e)

    def handle_queued_event(self) -> None:
        event = self._pending_event
        if event is not None:
            self._pending_event = None
            self._dispatch(event)


from . import source_registry  # noqa: E402 — registration side-effect
source_registry.register("smtc", MediaEventSource)
