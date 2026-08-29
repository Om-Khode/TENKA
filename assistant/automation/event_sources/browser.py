"""
browser.py — browser events, from the extension into the event bus.

A third `EventSource` alongside `window.py` and `media.py`. No new mechanism:
the bus already knows how to load monitors, evaluate conditions and fire
actions, and this only has to produce events that look like the ones it already
handles.

## The shape has to match what `check_dispatch` compares

`event_bus.check_dispatch` matches on `event_type`, then filters on
`source_app` through `normalize_process_name`, then dedups on
`source_app|window_title`. So every event here carries all three, and:

  - `source_app` is the **browser name** — `firefox`, `chrome`, `edge`,
    `brave`. That is what a person means by "when I open my bank in Firefox",
    and it is what a filter written by the parse prompt will say. Using the
    site's hostname instead would be an app-specific notion of "app" and would
    make every filter a brand name.
  - `window_title` is the page title, falling back to the url. It is half the
    dedup key, so leaving it out would collapse every event from one browser
    into a single deduped stream and drop all but the first.

`base.normalize_process_name`'s docstring records what happens when the two
halves of that comparison disagree: a monitor is created, loads, reports active
and can never fire. No error anywhere — a filter that matches nothing looks
exactly like an event that never happened. That is the failure this file is
written against, and the test asserts an end-to-end match through the real
`check_dispatch` rather than a string comparison of its own devising.

## Names are prefixed here, not on the wire

The extension speaks `navigated`; the bus sees `browser_navigated`. Drover is
host-agnostic and its event names say nothing about a host or a subsystem;
prefixing is this side's business, and doing it at the boundary keeps the
protocol neutral without making monitor names ambiguous.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

logger = logging.getLogger("event_bus.browser")

#: Wire name -> the bus's event_type. Prefixed so a monitor name cannot collide
#: with an OS-level one, and enumerated rather than computed so an event the
#: extension invents does not silently become a new monitor type.
_EVENT_TYPES: dict[str, str] = {
    "tab_opened": "browser_tab_opened",
    "tab_closed": "browser_tab_closed",
    "navigated": "browser_navigated",
    "download_started": "browser_download_started",
    "download_finished": "browser_download_finished",
}


class BrowserEventSource:
    """Satisfies the `EventSource` protocol; fed by the extension connection."""

    name = "browser"
    event_types = frozenset(_EVENT_TYPES.values())

    def __init__(
        self,
        connection_getter: Callable[[], object | None],
        *,
        subscribe: Callable[[Callable], None] | None = None,
        unsubscribe: Callable[[Callable], None] | None = None,
    ) -> None:
        # A getter, not a connection: the extension disconnects and reconnects
        # on its own schedule, and a source holding one instance would keep
        # feeding from a socket that closed an hour ago.
        self._get_connection = connection_getter
        self._subscribe = subscribe
        self._unsubscribe = unsubscribe
        self._dispatch: Callable[[dict], None] | None = None
        self._subscribed_to: object | None = None

    # ─── EventSource ─────────────────────────────────────────────────────

    def start(self, dispatch_fn: Callable[[dict], None] | None = None, **kwargs) -> None:
        if dispatch_fn is not None:
            self._dispatch = dispatch_fn
        # Attach now for a connection that is already up, and subscribe so a
        # later one attaches the moment it arrives. Both halves are needed: the
        # extension may connect before or after the bus starts, and which it is
        # depends on how fast a browser launches.
        self._attach()
        if self._subscribe is not None:
            self._subscribe(self._on_connection)

    def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe(self._on_connection)
        self._detach()
        self._dispatch = None

    def _on_connection(self, _connection) -> None:
        """A new extension connected. Move the subscription onto it."""
        self._attach()

    # ─── Wiring ──────────────────────────────────────────────────────────

    def _attach(self) -> None:
        connection = self._get_connection()
        if connection is None:
            return
        if connection is self._subscribed_to:
            # Idempotent. `start()` is called again on every reconnect, and a
            # second subscription to the same connection would dispatch every
            # event twice -- which reads downstream as the user doing the thing
            # twice, not as a bug here.
            return
        self._detach()
        connection.on_event(self._on_wire_event)
        self._subscribed_to = connection

    def _detach(self) -> None:
        if self._subscribed_to is not None:
            remove = getattr(self._subscribed_to, "remove_event_callback", None)
            if remove is not None:
                remove(self._on_wire_event)
            self._subscribed_to = None

    def poll(self) -> None:
        """Re-attach if the connection changed. Cheap and idempotent.

        The bus has no notion of a source whose transport comes and goes, so
        rather than teach it one, this is safe to call whenever something else
        already runs.
        """
        self._attach()

    # ─── Translation ─────────────────────────────────────────────────────

    def _on_wire_event(self, frame: dict) -> None:
        if self._dispatch is None:
            # Dropped rather than buffered: an event with nowhere to go is not
            # worth holding, and `stop()` must actually stop.
            return

        wire_name = frame.get("event")
        event_type = _EVENT_TYPES.get(wire_name)
        if event_type is None:
            logger.debug(f"[event-monitor] ignoring unknown browser event {wire_name!r}")
            return

        connection = self._subscribed_to
        source_app = getattr(connection, "browser_name", "") or "browser"
        title = str(frame.get("title") or "")
        url = str(frame.get("url") or "")

        try:
            self._dispatch({
                "event_type": event_type,
                # Matched through normalize_process_name against the monitor's
                # source_filter. The browser name is what a person names when
                # they say "when I open my bank in Firefox".
                "source_app": source_app,
                # Half the dedup key. Without it every event from one browser
                # collapses into a single stream and all but the first is
                # dropped as a duplicate.
                "window_title": title or url,
                "url": url,
                "title": title,
                "tab_id": frame.get("tabId"),
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            # A throwing consumer must not take the socket's event path down.
            logger.warning(f"[event-monitor] browser dispatch raised: {type(e).__name__}: {e}")


from . import source_registry  # noqa: E402 — registration side-effect

source_registry.register("browser", BrowserEventSource)
