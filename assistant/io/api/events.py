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
import time
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

# How long one socket may hold the fan-out before it is judged dead.
#
# `_pump` delivers sequentially, so whatever this number is, it is also the
# worst-case delay every *other* client pays for one peer that stopped
# reading. Before this bound existed there was no worst case at all: lens 5 F3
# proved that a `send_json` which never resolves stops the pump returning to
# the top of its loop, so a healthy socket attached afterwards received
# nothing, ever.
#
# Two seconds, matching `_REVALIDATE_INTERVAL_SECONDS` and the telemetry
# sampler's cadence: the longest a frame may take to be accepted is the
# interval before the next one would have arrived anyway. It is also generous
# for the slow-but-working case this must not evict -- uvicorn resolves a send
# as soon as the frame reaches the transport's buffer, so two seconds means
# the peer's receive window has been full for that entire time, which a phone
# on a bad tunnel does not do and a client that stopped reading always does.
#
# The cost is paid once per bad socket, not once per frame: a timeout drops
# the socket, so it is never awaited again.
_SEND_TIMEOUT_SECONDS = 2.0

# How many sockets one device may hold at once, and how many the hub will
# hold in total.
#
# The arithmetic, because the number is the whole finding. `app.py`'s socket
# handler spends the shared `RateLimiter` budget on every handshake, keyed per
# device: `security._MAX_PER_WINDOW` = 120 attempts per `_WINDOW_SECONDS` = 60.
# Lens 5 F6's attack is not a burst -- the limiter already stops that -- it is
# *accumulation*: 120 successful handshakes a minute, none of them ever closed,
# is ~7,200 live entries an hour that `_pump` iterates on every published
# frame.
#
# So the cap has to be the binding constraint, not the limiter. At 8 per
# device the cap is reached after 8 of that minute's 120 permitted handshakes
# -- roughly four seconds of a device's budget -- and the remaining 112 are
# refused here rather than accumulating. Any value at or above 120 would leave
# the limiter as the only bound and change nothing.
#
# 8 is also comfortably above honest use: Studio in several browser tabs plus
# a phone plus a wall display is four or five, and each reload replaces its
# predecessor as the old socket's `detach()` runs.
#
# The global cap is 8 devices' worth. It bounds the fan-out `_pump` performs
# per frame, which matters because delivery is sequential and each send is
# bounded by `_SEND_TIMEOUT_SECONDS` -- 64 is the worst-case number of
# timeouts a single frame could pay, and each one is paid at most once because
# a timed-out socket is dropped.
#
# Every bound below is applied with `asyncio.timeout`, never
# `asyncio.wait_for`. On 3.11 `wait_for` swallows an *outer* cancellation
# whenever the awaited future happens to already be done
# (`asyncio/tasks.py:477-479`, `if fut.done(): return fut.result()`), which
# turned `stop()` into a hang: cancelling `_revalidate` while it sat in a
# bounded `close()` left the loop running and `await task` waiting on it
# forever. `asyncio.timeout` only converts the cancellation it raised itself.
_MAX_SOCKETS_PER_DEVICE = 8
_MAX_SOCKETS_TOTAL = 64

# How long a socket may go without traffic in either direction before the hub
# stops holding it open.
#
# Two minutes, against a telemetry sampler that publishes to every attached
# socket every `_interval` (2.0) seconds while at least one is attached: a
# healthy connection on a running daemon refreshes this sixty times over
# before it could expire, so this can only fire on a socket nothing is
# reaching -- the sampler stopped, the runtime is absent, or the peer is
# attached to a hub that has genuinely gone quiet.
#
# Inbound traffic counts too, via `note_activity()`. The receive loop lives in
# `app.py`'s socket handler, so that call is the hook rather than something
# this module can observe for itself.
_IDLE_TIMEOUT_SECONDS = 120.0

# The same bound on the courtesy close. Task 17's `_drop(close=True)` added a
# second awaited call inside the same loop, which widened the wedge by one
# more place to enter it -- the socket being closed here has already been
# judged dead or unauthorised, so the handshake is a courtesy and one second
# is more than it is owed.
_CLOSE_TIMEOUT_SECONDS = 1.0

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
#
# Absence from this table is a decision, not an oversight, and the two absent
# types are worth naming because one of them is a trap. `telemetry` carries
# cpu, ram, battery, active model and uptime -- facts about the machine, none
# of them anything the user said. `error` carries a field *also* called
# `detail` (`build_error_frame`), which is exactly the coincidence that gets a
# field classified per call site instead of once: its values are four protocol
# literals ("malformed frame", "unknown frame", "not permitted",
# "unauthorized") and they describe the socket, not the person using it. Both
# types still pass through `visible_frame` -- `app.py`'s `_safe_send` sends
# nothing that does not -- so if either ever grows a user-derived field, the
# fix is one row here and the enforcement is already wired.
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
        # socket -> the device that opened it, for the per-device cap. `None`
        # for a caller that attached without naming one (test scaffolding, and
        # any call site not yet passing it): such a socket is not attributable
        # to a device, so only the global cap can apply to it.
        self._device_of: dict[Any, str | None] = {}
        # socket -> `time.monotonic()` of the last traffic in either
        # direction. Wall time would let a clock change reap every socket at
        # once.
        self._last_active: dict[Any, float] = {}
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
                     viewer: Callable[[], frozenset[Capability] | None] | None = None,
                     *, device_id: str | None = None) -> bool:
        """Start delivering to `socket`. `False` if the hub refused it.

        `viewer` is re-consulted, not cached: it must answer "what may this
        connection see now", by whatever means its owner considers
        authoritative (`app.py` re-reads the vault). Returning `None` closes
        the socket. Left unset -- test scaffolding attaching a bare object --
        the socket is delivered to unconditionally and unfiltered, exactly as
        before this parameter existed.

        `device_id` names who is holding the socket, for the per-device cap.
        A caller that does not pass one gets the global cap only: a socket
        nobody attributed cannot be counted against a device without guessing
        which, and guessing wrong locks out a device that did nothing.
        """
        # Captured here too, not just in start(): a test harness (or a
        # deployment that never calls start() before the first connection)
        # still needs publish() to land on the loop actually running this
        # socket's pump, not on whatever loop happened to exist first.
        self._set_loop(asyncio.get_running_loop())
        # Refused *before* the socket joins `_sockets`, and closed rather than
        # left dangling: the handshake already succeeded upstream, so the only
        # honest answer this layer can still give is to end the connection.
        if not self._has_room_for(device_id):
            logger.warning(
                f"[API] refusing an event socket at the concurrency cap "
                f"(device={device_id!r}, attached={len(self._sockets)})")
            await self._close_quietly(socket)
            return False
        self._sockets[socket] = viewer
        self._device_of[socket] = device_id
        self._last_active[socket] = time.monotonic()
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())
        if self._revalidate_task is None:
            self._revalidate_task = asyncio.create_task(self._revalidate())
        if self._telemetry_task is None and self._runtime is not None:
            self._telemetry_task = asyncio.create_task(self._sample_telemetry())
        return True

    def _has_room_for(self, device_id: str | None) -> bool:
        if len(self._sockets) >= _MAX_SOCKETS_TOTAL:
            return False
        if device_id is None:
            return True
        held = sum(1 for owner in self._device_of.values() if owner == device_id)
        return held < _MAX_SOCKETS_PER_DEVICE

    def note_activity(self, socket) -> None:
        """Record that `socket` is alive, from the receive side.

        The hub sees only what it sends. A device that holds the socket open
        and talks -- an `abort` frame, anything a later protocol adds -- is
        active by any honest definition, and `app.py`'s receive loop is the
        only place that can say so. Silently ignores a socket that is not (or
        is no longer) attached, so a frame racing a `detach()` is not an error.
        """
        if socket in self._sockets:
            self._last_active[socket] = time.monotonic()

    def _forget(self, socket) -> None:
        """Drop every per-socket record together. A cap whose counter outlives
        the socket it counted locks a device out after eight page reloads."""
        self._sockets.pop(socket, None)
        self._device_of.pop(socket, None)
        self._last_active.pop(socket, None)

    async def _close_quietly(self, socket) -> None:
        try:
            async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
                await socket.close(code=1008)
        except Exception:
            pass

    async def detach(self, socket) -> None:
        self._forget(socket)
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
        self._forget(socket)
        if not close:
            return
        try:
            # Bounded, because this runs inside `_pump`'s own loop: a peer that
            # is not reading does not accept a close frame any faster than it
            # accepts a data frame, and an unbounded await here would wedge the
            # fan-out at the exact moment the hub was trying to shed the socket
            # causing it.
            async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
                await socket.close(code=1008)
        except Exception:
            # Already gone, half-closed, too slow to say goodbye (3.11's
            # `TimeoutError` is an ordinary `Exception`, so the bound above
            # lands here), or a stub in a test. The socket is out of
            # `_sockets` either way, which is the part that matters.
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
                    # Bounded. This one `await` is the whole fan-out's
                    # critical section: delivery is sequential, so a send
                    # that never resolves is not a slow client, it is every
                    # client. See `_SEND_TIMEOUT_SECONDS` for the number.
                    async with asyncio.timeout(_SEND_TIMEOUT_SECONDS):
                        await socket.send_json(visible_frame(event, grants))
                    self.note_activity(socket)
                except asyncio.TimeoutError:
                    # Dead, not slow: the peer has not accepted a frame for
                    # the whole window. Closed as well as forgotten, so the
                    # far end learns the stream ended instead of holding a
                    # socket nothing will ever write to again.
                    logger.info("[API] dropping an event socket that did not "
                                "accept a frame within the send timeout")
                    await self._drop(socket, close=True)
                except Exception:
                    await self._drop(socket, close=False)

    async def _revalidate(self) -> None:
        """Close sockets whose device stopped being authorised, or that have
        gone silent, on a timer.

        The per-frame check in `_pump` covers every socket a frame reaches.
        This covers the one it does not: a device that only listens sends
        nothing to be checked on, and a quiet assistant publishes nothing
        either, so revocation would otherwise wait on traffic that may never
        come. This is the outbound half of "an accepted socket is not a
        credential" -- without it, the loudest devices are cut off first and
        the silent ones last, which is exactly backwards.

        The idle sweep rides the same timer rather than a second task: both
        answer "should this socket still be attached?", and one loop over
        `_sockets` is cheaper than two running out of phase with each other.
        """
        while True:
            await asyncio.sleep(_REVALIDATE_INTERVAL_SECONDS)
            now = time.monotonic()
            for socket, viewer in list(self._sockets.items()):
                if now - self._last_active.get(socket, now) > _IDLE_TIMEOUT_SECONDS:
                    logger.info("[API] closing an event socket with no traffic "
                                "in either direction for the idle window")
                    await self._drop(socket, close=True)
                    continue
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
        self._device_of.clear()
        self._last_active.clear()
