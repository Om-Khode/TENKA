# assistant/io/api/events.py
"""One socket carrying every live frame.

Not one SSE stream per concern: EventSource cannot set an Authorization header,
abort needs a client-to-server channel, and one socket means one reconnect
story on the client.

Three frame types actually have a producer today:

  - `"status"` -- forwarded from `assistant.io.status_broadcaster` via
    `EventHub.publish_status()` (main.py subscribes that method, not
    `publish()` directly, once at startup: `status.subscribe(hub.publish_status)`).
    The connect-time frame `app.py` sends before any broadcaster event has
    fired is built by the same `build_status_frame()` used here, so it carries
    the identical key set -- a client never has to special-case "the first
    status frame" versus every later one. There is no separate frame type for
    task/step progress: `step` (`[n, total] | None`) and `tier` already ride
    as fields on `"status"` (`status_broadcaster.py`'s `set()`); a live task
    display should read those off `"status"`, not wait on a distinct frame
    type that nothing produces.
  - `"telemetry"` -- sampled from the runtime's `TelemetrySnapshot` every
    `_interval` seconds while >=1 socket is attached (`_sample_telemetry`
    below). Keys are `telemetry_body()`'s (`routes/system.py`, the same
    function `GET /v1/telemetry` uses) so the wire has one telemetry
    vocabulary, not one per transport.
  - `"error"` / `"ack"` -- built inline in `app.py`'s socket handler, in reply
    to a malformed/unrecognised client frame or a client's own
    `{"type": "abort"}`. Not hub-produced, so not covered by the frame
    builders below.

`"toast"` is reserved, not produced: a one-off notification is a different
shape of thing than a phase transition, so the type name is worth keeping for
whenever something actually emits one -- but nothing in `assistant/` does
today. Treat it as reserved-and-unproduced; do not build a client that waits
on it.

main.py subscribes the hub's `publish_status()` to status_broadcaster once,
at startup, so every socket the hub ever attaches receives its status frames;
the telemetry sampler runs only while at least one socket is attached.

`publish()` is the one method this module promises is safe to call from any
thread, not just the event loop's. `status_broadcaster` calls it from
synchronous code that may be running on a worker thread, and
`asyncio.Queue.put_nowait` is documented as safe only from the loop's own
thread -- calling it directly from elsewhere races the loop's internal
bookkeeping. `publish()` therefore hands the enqueue off to the loop captured
at `start()`/`attach()` time via `call_soon_threadsafe`, which is the one
asyncio primitive built for exactly this handoff. Before any loop has been
captured (nothing has started or attached yet -- exactly the window between
main.py wiring the subscription and the daemon's first socket, during which
status_broadcaster can already be firing from arbitrary worker threads), the
event is buffered rather than touched at all: there is no thread on which
enqueueing directly would be safe, so there is no such fallback. Once a loop
is captured, the buffer drains onto it the same way every later publish()
does.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any, Callable

from .routes.system import telemetry_body
from .vault import Capability

logger = logging.getLogger(__name__)

# How often an attached socket is re-checked when no frame is flowing. The
# per-frame check below makes revocation immediate for anything actually
# delivered; this is what covers the device that only *listens* -- it never
# sends a frame to be checked on, and a quiet assistant publishes none either,
# so without this a revoked watcher could sit attached indefinitely simply by
# being lucky about timing. Two seconds matches the telemetry sampler's own
# cadence: the same order as "the next thing that would have reached it".
_REVALIDATE_INTERVAL_SECONDS = 2.0

# Fields on a wire frame that carry what the *user* said, as opposed to facts
# about the assistant, and the grant it therefore takes to see them.
#
# `detail` is the whole list today, and it earns its place by its producers
# rather than by its name: `status_broadcaster.set(detail=...)` is called with
# `goal`, `target`, the text about to be typed, a URL, an app name, a search
# query and the sentence TENKA is about to speak, from `actions/`,
# `code_executor/` and `io/audio/`. Some producers pass a fixed literal
# ("synthesizing", "running the code") instead -- but the field has no schema
# that separates the two, and any allow-list of "safe" details would be the
# app-specific hardcoding this project does not do. So the field is classified
# by its worst producer, which is the only classification that stays true when
# somebody adds the next one.
#
# RECALL, specifically. `policy.py` argues that grant carries "the entire
# knowledge graph and every transcript", and that excluding SCREEN while
# admitting stored data "withheld the photograph and shipped the description
# of it". The first forty characters of what she was asked to do is that
# description, arriving live. A device paired to watch -- and the Cloudflare
# `quick` tunnel, whose {OBSERVE} ceiling exists precisely to keep the user's
# content away from an intermediary that reads the plaintext -- keeps
# everything that is about *her*: `phase`, `step`, `tier`, `cursorFollows`,
# `ts`. Nothing approved as observable is lost, including which model is
# active: that rides `telemetry_body()` (the `"telemetry"` frame and
# `GET /v1/telemetry`) and `GET /v1/status`, all of them OBSERVE-gated
# already, so it reaches a watching device by the routes that were always
# meant to carry it.
_USER_CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    "status": ("detail",),
}
_USER_CONTENT_CAPABILITY = Capability.RECALL


def visible_frame(frame: dict, grants: frozenset[Capability] | None) -> dict:
    """`frame` as a device holding `grants` may see it.

    Blanked, never dropped: `test_the_connect_frame_and_a_real_status_frame_
    share_one_shape` and the client that relies on it both require every
    `"status"` frame to carry the identical key set, so a withheld field
    stays present as `""`. A client cannot tell "she was given no detail"
    from "you may not read it" -- which is the correct answer to give a
    device that may not read it.

    `grants` of `None` means "no capability model applies here" -- a caller
    that attached a socket without one (test scaffolding). Unfiltered, as it
    was before this existed.
    """
    if grants is None:
        return frame
    if _USER_CONTENT_CAPABILITY in grants:
        return frame
    fields = _USER_CONTENT_FIELDS.get(frame.get("type", ""), ())
    if not fields:
        return frame
    return {**frame, **{name: "" for name in fields if name in frame}}


# ─── frame builders ─────────────────────────────────────────────────────────
# Pure dict-in/dict-out functions, deliberately: this module may not import
# `assistant.io.status_broadcaster` (io/api sits one layer below `assistant.io`
# and may only reach `assistant.core`/`assistant.config`/third-party), so a
# broadcaster event is accepted here only as data -- a dict shaped the way
# `status_broadcaster.py`'s `set()` builds one -- never via the module that
# builds it. main.py is what actually bridges the two modules together.

def build_status_frame(*, phase: str, detail: str = "",
                        v: int | None = None,
                        cursor_follows: bool | None = None,
                        step: list[int] | None = None,
                        tier: str | None = None,
                        ts: float | None = None) -> dict:
    """The one `"status"` wire shape, camelCased. Shared by the connect-time
    synthetic frame (`app.py`, before any broadcaster event has ever fired)
    and every real one (`status_frame_from_broadcaster_event` below) so both
    carry exactly the same key set -- whatever the caller doesn't know yet is
    passed through as `None` (-> JSON `null`), never omitted.
    """
    return {
        "v": v,
        "type": "status",
        "phase": phase,
        "detail": detail,
        "cursorFollows": cursor_follows,
        "step": step,
        "tier": tier,
        "ts": ts,
    }


def status_frame_from_broadcaster_event(event: dict) -> dict:
    """Translate a `status_broadcaster.py`-shaped event dict (`v`, `type`,
    `phase`, `detail`, `cursor_follows`, `step`, `tier`, `ts`) into this
    socket's wire frame -- camelCasing `cursor_follows` and dropping nothing.
    Takes a plain dict, not a broadcaster import; see the module note above.
    """
    return build_status_frame(
        v=event.get("v"),
        phase=event.get("phase", ""),
        detail=event.get("detail", ""),
        cursor_follows=event.get("cursor_follows"),
        step=event.get("step"),
        tier=event.get("tier"),
        ts=event.get("ts"),
    )


def telemetry_frame(snapshot) -> dict:
    """The one `"telemetry"` wire shape -- `telemetry_body()`'s keys plus the
    frame envelope's `type`, so this can never drift from what
    `GET /v1/telemetry` reports for the same snapshot.

    `telemetry_body()` now returns a `TelemetryPayload` model (the typed-
    response rework), not a plain dict -- `model_dump(by_alias=True)` is the
    one place that has to change to keep spreading its keys; the camelCase
    key set itself is unaffected.
    """
    return {"type": "telemetry", **telemetry_body(snapshot).model_dump(by_alias=True)}


def build_error_frame(detail: str) -> dict:
    """The `"error"` reply app.py sends for a malformed or unrecognised
    client frame. Routed through a builder -- not a literal dict at the
    call site -- so the socket's one snake_case sweep (see the module
    docstring and `tests/test_api_events.py`) covers it the same way it
    covers every other frame the socket can emit, even though `EventHub`
    itself never produces this one.
    """
    return {"type": "error", "detail": detail}


def build_ack_frame(of: str) -> dict:
    """The `"ack"` reply app.py sends after acting on a client frame (only
    `{"type": "abort"}` today). Same reasoning as `build_error_frame`.
    """
    return {"type": "ack", "of": of}


class EventHub:
    """Fan-out with a live authorisation check attached to each socket.

    A socket is not a credential that was checked once. `attach()` takes a
    `viewer` callable -- "what may this connection see *right now*?" -- and
    the hub consults it before every frame it delivers and on a timer while
    nothing is flowing. `None` from that callable means the connection is no
    longer authorised at all, and the hub closes it rather than delivering.
    See `app.py`'s socket handler for the implementation of one, and the
    revocation argument for why an accepted socket cannot be trusted for its
    own lifetime.
    """

    def __init__(self) -> None:
        # socket -> viewer callable (or None when a caller attached without
        # one). A dict rather than the set this used to be, because "who is
        # on the other end of this socket" is now something the hub has to be
        # able to answer per socket, not just per fan-out.
        self._sockets: dict[Any, Callable[[], frozenset[Capability] | None] | None] = {}
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1_000)
        self._telemetry_task: asyncio.Task | None = None
        self._revalidate_task: asyncio.Task | None = None
        self._pump_task: asyncio.Task | None = None
        self._runtime = None
        self._interval = 2.0
        self._loop: asyncio.AbstractEventLoop | None = None
        # Guards `_loop` and `_pending` only -- both are touched from
        # publish()'s arbitrary caller threads as well as the loop thread
        # that runs attach()/start(). Nothing else in this class crosses
        # threads, so nothing else needs it.
        self._state_lock = threading.Lock()
        self._pending: list[dict] = []

    # ─── membership ─────────────────────────────────────────────────────
    def subscriber_count(self) -> int:
        return len(self._sockets)

    def telemetry_running(self) -> bool:
        return self._telemetry_task is not None and not self._telemetry_task.done()

    async def attach(self, socket,
                     viewer: Callable[[], frozenset[Capability] | None] | None = None) -> None:
        """Start delivering to `socket`.

        `viewer` is re-consulted, not cached: it must answer "what may this
        connection see now", by whatever means its owner considers
        authoritative (`app.py` re-reads the vault). Returning `None` closes
        the socket. Left unset -- test scaffolding attaching a bare object --
        the socket is delivered to unconditionally and unfiltered, exactly as
        before this parameter existed.
        """
        # Captured here too, not just in start(): a test harness (or a
        # deployment that never calls start() before the first connection)
        # still needs publish() to land on the loop actually running this
        # socket's pump, not on whatever loop happened to exist first.
        self._set_loop(asyncio.get_running_loop())
        self._sockets[socket] = viewer
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())
        if self._revalidate_task is None:
            self._revalidate_task = asyncio.create_task(self._revalidate())
        if self._telemetry_task is None and self._runtime is not None:
            self._telemetry_task = asyncio.create_task(self._sample_telemetry())

    async def detach(self, socket) -> None:
        self._sockets.pop(socket, None)
        if not self._sockets:
            for name in ("_telemetry_task", "_revalidate_task"):
                task = getattr(self, name)
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    setattr(self, name, None)

    # ─── fan-out ────────────────────────────────────────────────────────
    def publish(self, event: dict) -> None:
        """Non-blocking. Safe from sync code on any thread (status_broadcaster).

        Sends `event` verbatim -- no translation happens here. Anything that
        needs shaping before it reaches a socket (a status_broadcaster event,
        today) goes through a dedicated method like `publish_status()`
        instead, which translates and then calls this.

        Buffers rather than enqueues while no loop is known yet -- see the
        module docstring for why a direct enqueue here is never safe.
        """
        with self._state_lock:
            loop = self._loop
            if loop is None:
                self._pending.append(event)
                return
        loop.call_soon_threadsafe(self._enqueue, event)

    def publish_status(self, event: dict) -> None:
        """The bridge point for `status_broadcaster`: main.py subscribes this
        method (`status.subscribe(hub.publish_status)`), not `publish()`
        directly, so every broadcaster event is translated to this socket's
        camelCase wire shape (`status_frame_from_broadcaster_event`) before
        anything reaches a browser. Safe from any thread, same as `publish()`.
        """
        self.publish(status_frame_from_broadcaster_event(event))

    def _set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the loop once, then flush anything published before it existed."""
        with self._state_lock:
            if self._loop is not None:
                return
            self._loop = loop
            pending, self._pending = self._pending, []
        for event in pending:
            loop.call_soon_threadsafe(self._enqueue, event)

    def _enqueue(self, event: dict) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("[API] event queue full; dropping a frame")

    async def _drop(self, socket, *, close: bool) -> None:
        """Forget a socket, optionally closing it first.

        Never `detach()`: this is called from `_pump` and `_revalidate`, and
        `detach()` awaits the cancellation of the revalidate task, which from
        inside that task is a wait on itself.
        """
        self._sockets.pop(socket, None)
        if not close:
            return
        try:
            await socket.close(code=1008)
        except Exception:
            # Already gone, half-closed, or a stub in a test. The socket is
            # out of `_sockets` either way, which is the part that matters.
            pass

    async def _pump(self) -> None:
        while True:
            event = await self._queue.get()
            for socket, viewer in list(self._sockets.items()):
                # Authorisation, re-asked per frame. The handshake's answer is
                # never reused: a device revoked a moment ago must not receive
                # this frame, and `TokenVault.verify()` re-reading disk on
                # every call is the only reason the one-year cookie is
                # defensible in the first place. HTTP already pays exactly
                # this cost per request; the socket's frame rate is set by the
                # assistant's own work, not by the client, so it cannot be
                # driven up from the other end.
                grants = None
                if viewer is not None:
                    try:
                        grants = viewer()
                    except Exception:
                        grants = None
                    if grants is None:
                        logger.info("[API] closing an event socket whose "
                                    "device is no longer authorised")
                        await self._drop(socket, close=True)
                        continue
                try:
                    await socket.send_json(visible_frame(event, grants))
                except Exception:
                    await self._drop(socket, close=False)

    async def _revalidate(self) -> None:
        """Close sockets whose device stopped being authorised, on a timer.

        The per-frame check in `_pump` covers every socket a frame reaches.
        This covers the one it does not: a device that only listens sends
        nothing to be checked on, and a quiet assistant publishes nothing
        either, so revocation would otherwise wait on traffic that may never
        come. This is the outbound half of "an accepted socket is not a
        credential" -- without it, the loudest devices are cut off first and
        the silent ones last, which is exactly backwards.
        """
        while True:
            await asyncio.sleep(_REVALIDATE_INTERVAL_SECONDS)
            for socket, viewer in list(self._sockets.items()):
                if viewer is None:
                    continue
                try:
                    still = viewer()
                except Exception:
                    still = None
                if still is None:
                    logger.info("[API] closing an idle event socket whose "
                                "device is no longer authorised")
                    await self._drop(socket, close=True)

    async def _sample_telemetry(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if not self._sockets or self._runtime is None:
                continue
            try:
                snapshot = await self._runtime.system.telemetry()
            except Exception as exc:
                logger.debug(f"[API] telemetry sample failed: {exc}")
                continue
            self.publish(telemetry_frame(snapshot))

    # ─── lifecycle ──────────────────────────────────────────────────────
    async def start(self, runtime, interval_seconds: float = 2.0) -> None:
        self._runtime = runtime
        self._interval = interval_seconds
        self._set_loop(asyncio.get_running_loop())
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        for task in (self._telemetry_task, self._revalidate_task, self._pump_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._telemetry_task = None
        self._revalidate_task = None
        self._pump_task = None
        self._sockets.clear()
