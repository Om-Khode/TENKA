"""WinEventHook event source for.

Subscribes to EVENT_SYSTEM_FOREGROUND and EVENT_OBJECT_NAMECHANGE via
SetWinEventHook. Callbacks fire on the message pump thread directly
(WINEVENT_OUTOFCONTEXT flag).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import os
from datetime import datetime
from typing import Callable

logger = logging.getLogger("event_bus.window")

# Win32 constants
EVENT_SYSTEM_FOREGROUND = 0x0003
EVENT_OBJECT_NAMECHANGE = 0x800C
WINEVENT_OUTOFCONTEXT = 0x0000
OBJID_WINDOW = 0

WinEventProcType = ctypes.WINFUNCTYPE(
    None,
    ctypes.wintypes.HANDLE,   # hWinEventHook
    ctypes.wintypes.DWORD,    # event
    ctypes.wintypes.HWND,     # hwnd
    ctypes.wintypes.LONG,     # idObject
    ctypes.wintypes.LONG,     # idChild
    ctypes.wintypes.DWORD,    # idEventThread
    ctypes.wintypes.DWORD,    # dwmsEventTime
)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def _get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _get_process_name(hwnd: int) -> str:
    pid = ctypes.wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == 0:
        return ""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.wintypes.DWORD(260)
        kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return os.path.basename(buf.value).replace(".exe", "")
    finally:
        kernel32.CloseHandle(handle)


def _get_foreground_hwnd() -> int:
    return user32.GetForegroundWindow()


class WindowEventSource:
    name: str = "window"
    event_types: frozenset[str] = frozenset({"window_focus", "window_title"})

    def __init__(self, dispatch_fn: Callable[[dict], None]) -> None:
        self._dispatch = dispatch_fn
        self._hook: ctypes.wintypes.HANDLE | None = None
        self._last_app: str = ""
        self._last_title: str = ""
        self._callback = WinEventProcType(self._win_event_proc)

    def start(self, dispatch_fn: Callable[[dict], None] | None = None, **kwargs) -> None:
        """Protocol-compatible start; delegates to register()."""
        self.register()

    def stop(self) -> None:
        """Protocol-compatible stop; delegates to unregister()."""
        self.unregister()

    def register(self) -> bool:
        self._hook = user32.SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_OBJECT_NAMECHANGE,
            0,
            self._callback,
            0, 0,
            WINEVENT_OUTOFCONTEXT,
        )
        if not self._hook:
            logger.error("[event-monitor] Failed to register WinEventHook")
            return False
        logger.info("[event-monitor] WinEventHook registered")
        return True

    def unregister(self) -> None:
        if self._hook:
            user32.UnhookWinEvent(self._hook)
            self._hook = None
            logger.info("[event-monitor] WinEventHook unregistered")

    def _win_event_proc(
        self, hook, event, hwnd, id_object, id_child, event_thread, event_time,
    ) -> None:
        try:
            if event == EVENT_SYSTEM_FOREGROUND:
                title = _get_window_title(hwnd)
                process = _get_process_name(hwnd)
                if not process:
                    return
                self._dispatch({
                    "event_type": "window_focus",
                    "source_app": process,
                    "window_title": title,
                    "prev_app": self._last_app,
                    "prev_title": self._last_title,
                    "timestamp": datetime.now().isoformat(),
                })
                self._last_app = process
                self._last_title = title

            elif event == EVENT_OBJECT_NAMECHANGE and id_object == OBJID_WINDOW:
                fg = _get_foreground_hwnd()
                if hwnd != fg:
                    return
                title = _get_window_title(hwnd)
                process = _get_process_name(hwnd)
                if not process:
                    return
                self._dispatch({
                    "event_type": "window_title",
                    "source_app": process,
                    "window_title": title,
                    "timestamp": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.debug("[event-monitor] WinEvent callback error: %s", e)


from . import source_registry  # noqa: E402 — registration side-effect
source_registry.register("window", WindowEventSource)
