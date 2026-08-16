"""
main.py — Entry point for the TENKA Voice Assistant.

This is the pipeline orchestrator that ties everything together:
  1. Starts the TCP bridge (for Unity communication)
  2. Listens for events from Unity OR keyboard input
  3. Runs the pipeline: STT → Intent → Policy → Action/LLM → TTS
  4. Sends animation/expression commands back to Unity

Run with:
    python -m assistant.main

Or via start_assistant.bat
"""

import asyncio
import logging
import pathlib
import re
import signal
import sys
import subprocess
import os
import threading
import queue

# Must be set before ANY speechbrain import — speechbrain's get_logger() reads this
# env var and calls logger.setLevel() during model load, overriding any later setLevel calls.
os.environ.setdefault("SB_LOG_LEVEL", "WARNING")
import warnings
import time as _time

from . import config
from .io.unity_bridge import UnityBridge, NullBridge
from .io.audio.stt import recorder, transcribe, record_until_silence, calibrate_noise_floor
from .intent import detect_intent
from .policy import evaluate as evaluate_policy
from .actions import execute as execute_action
from .io.audio import tts
from . import llm
from . import memory
from . import personality
from . import preferences
from . import shortcuts
from . import procedures
from . import settings
from . import slash_commands
from . import actions as _actions_module
from . import regex_router
from . import proactive
from . import reminders
from .io.audio.wake_word import WakeWordListener
from .io import messaging_bridge
from . import recording
from .io.audio import speaker_verify
from . import telemetry as _telemetry
from . import knowledge_graph
from datetime import datetime as _dt
from .core.abort import abort
from .core.capabilities import Capability
from .core.redact import redact_secrets
from .io.esc_monitor import esc_monitor
from .io.status_broadcaster import status, StatusPhase
from .overlay_manager import overlay_manager
from .io.audio.streaming import stop_streaming

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

if config.DEBUG_LOG:
    _debug_file_handler = logging.FileHandler("assistant/debug.log", mode="w", encoding="utf-8")
    _debug_file_handler.setLevel(logging.DEBUG)
    _debug_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(_debug_file_handler)

def _silence_third_party_loggers():
    """
    Silence noisy third-party loggers. Called at module level AND after model
    loading, because speechbrain's get_logger() and sentence_transformers both
    call setLevel(INFO) during model load, overriding earlier settings.
    """
    # speechbrain startup chatter (SB_LOG_LEVEL env var handles most of it,
    # but set explicitly too as belt-and-suspenders)
    for _name in (
        "speechbrain",
        "speechbrain.utils.quirks",
        "speechbrain.utils.fetching",
        "speechbrain.utils.parameter_transfer",
    ):
        logging.getLogger(_name).setLevel(logging.WARNING)
    # sentence-transformers device/load notices
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    # Gemini SDK AFC info
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)
    # transformers unexpected-key / position_ids
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    # whatsmeow group-decrypt failures and status noise — unfixable, spam constantly
    logging.getLogger("whatsmeow.Client").setLevel(logging.ERROR)
    logging.getLogger("whatsmeow.Client.Socket").setLevel(logging.ERROR)


_silence_third_party_loggers()

# Suppress position_ids / UNEXPECTED key warnings at the Python warnings level too
# (some libraries use warnings.warn instead of logging)
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

logger = logging.getLogger("main")

# ─── Wellbeing Safeguard ────────────────────────────────────────────────────
_WELLBEING_CHECKINS = [
    "By the way — talked to anyone human today? Just checking.",
    "Hey — when's the last time you talked to a real person? No judgement, just checking.",
    "Random thought — have you talked to anyone human today?",
    "Quick check-in: you've been here a while. Anyone human in the mix today?",
]
_WELLBEING_INTERVAL = 50

# ─── Whisper.cpp Server Auto-Start ────────────────────────────────────────────

_whisper_process = None


def _start_whisper_cpp_server():
    """
    Auto-start the whisper.cpp HTTP server if using the whisper_cpp backend
    and the executable exists.
    """
    global _whisper_process

    if config.STT_BACKEND != "whisper_cpp":
        return

    exe = config.WHISPER_CPP_EXE
    model = config.WHISPER_CPP_MODEL

    if not exe.exists():
        logger.warning(
            f"whisper-server.exe not found at {exe}. "
            "Please start the whisper.cpp server manually, or switch to faster_whisper."
        )
        return

    if not model.exists():
        logger.warning(f"Whisper model not found at {model}.")
        return

    try:
        _whisper_process = subprocess.Popen(
            [str(exe), "-m", str(model), "--port", str(config.WHISPER_CPP_PORT)],
            cwd=str(config.WHISPER_CPP_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        logger.info(
            f"whisper.cpp server started on port {config.WHISPER_CPP_PORT} "
            f"(PID: {_whisper_process.pid})"
        )
    except Exception as e:
        logger.error(f"Failed to start whisper.cpp server: {e}")


def _stop_whisper_cpp_server():
    """Stop the auto-started whisper.cpp server."""
    global _whisper_process
    if _whisper_process and _whisper_process.poll() is None:
        try:
            _whisper_process.terminate()
            _whisper_process.wait(timeout=5)
            logger.info("whisper.cpp server stopped")
        except Exception as e:
            logger.warning(f"Error stopping whisper.cpp: {e}")
    _whisper_process = None


# ─── Chat Input Queue ─────────────────────────────────────────────────────────

_input_queue = queue.Queue()
_shutdown_event = asyncio.Event()

_turn_counter: int = 0
_topic_tracker = None


def _get_topic_tracker():
    global _topic_tracker
    if _topic_tracker is None:
        from . import topic_tracker as _tt
        _topic_tracker = _tt.TopicTracker()
        # knowledge-graph Session 2 Issue 1: register as active so other domain modules
        # (knowledge_graph) can read the same tracker without an injected dep.
        _tt.set_active(_topic_tracker)
    return _topic_tracker


# ─── Incoming Message Notification State ─────────────────────────────────

_notification_debounce_buffer: dict[tuple[str, str], dict] = {}
# Key: (service, sender_phone)
# Value: {"sender_name": str, "messages": [{"text": ..., "timestamp": ...}], 
#         "first_seen": float, "chat_type": str}

def _chat_input_loop(input_queue):
    """Read text input from terminal and push to shared queue.

    Delegates to chat_input.chat_input_loop, which uses prompt_toolkit for
    slash-command autocomplete + history when available, else falls back to
    bare input().
    """
    from .io import chat_input
    chat_input.chat_input_loop(input_queue)


# ─── Studio dispatch ──────────────────────────────────────────────────────

class _StudioDispatch:
    """The only path from an HTTP request into the pipeline.

    One turn at a time: a second concurrent submit is refused rather than
    queued, because a browser tab left open should not be able to drive her
    minutes after its owner walked away.

    A plain bool, not an asyncio.Lock: submit() enqueues onto _input_queue
    and returns immediately -- there is no `await` anywhere between checking
    and setting the flag, so a lock bought nothing an equivalent bool
    couldn't (an `async with` around a body with no suspension point can
    never actually be observed "held" by a second coroutine; see the
    original version this replaced). What actually needs to span "one turn"
    is not the enqueue, which is instant, but the turn itself, which the
    queue consumer processes asynchronously afterwards -- `mark_done()`,
    called from process_text_from_queue's own `finally` block once a
    "studio"-sourced turn has genuinely finished, is what the lock never did:
    hold the busy state for the turn's real duration.
    """

    def __init__(self) -> None:
        self._busy = False
        self._counter = 0

    @property
    def busy(self) -> bool:
        return self._busy

    async def submit(self, text: str,
                     grants: "frozenset[Capability]") -> tuple[str, str, bool, str]:
        if self._busy:
            return ("", "", False, "busy")
        self._busy = True
        from . import session as session_mod
        self._counter += 1
        turn_id = f"studio-{self._counter}"
        # A 3-tuple, not the local sources' 2-tuple: the grant set has to
        # travel with the turn, because the turn runs later on the consumer's
        # task and the request that authorised it is gone by then.
        # `_grants_for_item` is what reads it back, and it reads the third
        # slot by source -- for a local item that slot is stt_ms.
        _input_queue.put(("studio", text, grants))
        return (turn_id, session_mod.get_current_session_id(), True, "")

    def mark_done(self) -> None:
        """Called once, from process_text_from_queue's `finally` block, for
        every turn whose source was "studio" -- success, failure, or an
        early return partway through all reach it. Clearing the flag any
        earlier (e.g. right after the enqueue) is the exact bug this
        replaces."""
        self._busy = False

    async def abort(self) -> bool:
        abort.request_abort("studio")
        return True


_studio_dispatch: "_StudioDispatch | None" = None
# Module-level for the same reason `_studio_vault` is: process_text_from_queue
# needs to reach the one dispatch instance _start_studio_daemon() built, to
# call mark_done() once a "studio"-sourced turn finishes, and it has no other
# handle on it (the instance is built and handed to build_studio_runtime()
# inside that function's own local scope).

_studio_vault: "TokenVault | None" = None
# Module-level, not returned: _start_studio_daemon()'s return type
# (`asyncio.Task | None`) is pinned by tests/test_api_server_lifecycle.py,
# which asserts on it directly (`result is None`, `task is not None`,
# `await task`). The vault has to be reachable from wherever shutdown runs
# too -- a second local variable in a different function's scope can't do
# that -- so it lives here instead of riding the return value.

_studio_pair_store: "PairCodeStore | None" = None
# Same reachability need as `_studio_vault`, for a stronger reason:
# `PairCodeStore` is deliberately in-memory only (pairing.py's docstring --
# a three-minute code has no reason to survive a restart, and not
# persisting it means there is no file to steal), so there is no
# file-backed equivalent of `TokenVault(config.SANDBOX_DIR)` a caller could
# reconstruct from outside the running app the way `_studio_vault()` in
# slash_commands.py does. This module-level reference IS the only handle on
# the one store `POST /v1/pair` actually consults. `/studio pair`
# (slash_commands.py) reads this global rather than building its own
# `PairCodeStore()` -- doing that would mint a code into an object the
# pairing route can never see: a code that looks perfectly valid on the
# console and can never be redeemed. Invariant this global is meant to hold
# at every instant: non-`None` if and only if a Studio app is actually
# serving `POST /v1/pair` right now. `_start_studio_daemon()` only assigns
# it *after* `serve_studio_api()` has returned without raising (never
# before, even though the `PairCodeStore` itself has to exist earlier to be
# threaded through as an argument -- see that function's own comment), and
# `_stop_studio_daemon()` clears it back to `None` unconditionally, first
# thing, on every one of its exit paths. `None` here means no daemon has
# ever successfully started (or one that did has since stopped), and the
# slash command must say so plainly rather than mint anyway.


_VENDORED_UI_ZIP = pathlib.Path(__file__).resolve().parent / "io" / "api" / "studio_ui.zip"
# The bundle `tools/package_studio_ui.py` writes on a release. A module-level
# constant rather than a path built inside the function below, so that a test
# can point it somewhere else -- "there is no vendored bundle" is a state this
# has to behave well in and one that cannot otherwise be reached from a
# checkout that has one.


def _resolve_studio_ui() -> "UiBundle | None":
    """Decide which Studio build the daemon serves, if any.

    This function exists because of a layering rule, not a design preference.
    `UiBundle` knows how to *open* a bundle and deliberately knows nothing
    about where one lives: `io/api` may not import `config` (a single lazy
    import there would drag in `assistant.llm` and `assistant.storage` and
    break two import-linter contracts -- see the closing comment in ui.py). So
    the path decision is main.py's, and the opened bundle is injected the same
    way `runtime`, `vault` and `origins` already are.

    Two sources, in the order `UiBundle.open` applies them:

    * `studio_ui_path`, a developer's own `next build` output. It wins so that
      iterating on Studio needs no re-vendoring.
    * the vendored `assistant/io/api/studio_ui.zip`, written by
      `tools/package_studio_ui.py` on a release.

    Neither one is required. A daemon with no bundle serves its API and no
    pages -- which is the correct outcome for a fork that has not vendored a
    build, and the only safe one for a bundle that turns out to be unreadable.
    The UI is a convenience; the API is the product.
    """
    from pathlib import Path

    from .io.api.ui import MARKER_NAME, UiBundle

    configured = (config.STUDIO_UI_PATH or "").strip()
    dir_path = Path(configured).expanduser() if configured else None
    zip_path = _VENDORED_UI_ZIP

    if dir_path is not None and not (dir_path / MARKER_NAME).is_file():
        # Said out loud, because the failure is otherwise invisible: the
        # directory silently loses to the vendored zip and the developer sees a
        # stale UI with no explanation. A raw export has no marker in it --
        # `next build` does not write one -- so this is the expected state of a
        # freshly built `out/`, not an exotic one.
        logger.warning(
            f"[UI] studio_ui_path is set to {dir_path} but there is no "
            f"{MARKER_NAME} in it, so it cannot be served; falling back to the "
            f"vendored bundle. Run tools/package_studio_ui.py --marker to write "
            f"one into that directory.")

    bundle = UiBundle.open(zip_path=zip_path if zip_path.is_file() else None,
                           dir_path=dir_path)
    if bundle is None:
        logger.warning(
            "[UI] no Studio UI bundle -- the daemon will serve its API and "
            "nothing at `/`. Set studio_ui_path to a local export, or run "
            "tools/package_studio_ui.py to vendor one.")
    return bundle


async def _start_studio_daemon() -> "asyncio.Task | None":
    """Build and start the Studio daemon. Returns the running task, or
    None if anything failed -- the daemon is an optional side channel and
    must never stop the assistant itself from booting.

    Split out of async_main() so a test can drive this exact sequence
    (e.g. forcing serve() to fail) without booting the assistant.
    """
    global _studio_dispatch, _studio_vault, _studio_pair_store
    try:
        from .actions.studio_runtime import build_studio_runtime
        from .io.api.events import EventHub
        from .io.api.pairing import PairCodeStore
        from .io.api.server import serve as serve_studio_api
        from .io.api.vault import Capability, TokenVault

        _studio_vault = TokenVault(config.SANDBOX_DIR)
        if not _studio_vault.devices():
            # `frozenset(Capability)` is the same enum-derived spelling that
            # Milestone 6a.5 deleted from policy.py's ceilings -- and here it
            # is correct. This is the operator's own desktop Studio, on the
            # loopback listener, and it should hold every capability
            # including any added later; that is what "the person is at the
            # machine" means. Do not "fix" it by applying policy.py's lesson
            # uniformly: an explicit literal here would silently strip the
            # operator's own client of EXECUTE the next time the enum grows,
            # and it would read as a bug, not as a policy change. The ceiling
            # is where transports are decided; this is a device.
            _studio_token = _studio_vault.issue("studio", frozenset(Capability))
            # The raw token goes to stdout ONLY -- a browser can't read a
            # file, so this is the one and only time the operator can
            # copy it into Studio's connect screen. The log line names
            # the event, never the value: DEBUG_LOG defaults to true on a
            # fresh install, so anything logged here reaches debug.log in
            # plaintext (KI-12). redact_secrets() elsewhere in this
            # function is a second line of defence, not the reason the
            # token stays out -- it never appears in a string handed to
            # logger in the first place.
            logger.info("[API] Studio device token issued -- printed to "
                        "the console once, not logged.")
            print(f"\n  TENKA Studio token: {_studio_token}\n")

        # hub is built, but NOT subscribed to status_broadcaster yet:
        # StatusBroadcaster.subscribe() has no matching unsubscribe, so a
        # subscription made before serve() can still fail (e.g. an
        # eagerly-checked bad TENKA_SECRET raising inside create_app())
        # would outlive the failed daemon -- every future status.set()
        # call, which fires on every phase transition, would keep
        # appending to a hub nothing ever drains, for the rest of the
        # process, on a machine where the daemon is off. Subscribing only
        # after serve() has returned without raising means a startup
        # failure leaves no trace to leak.
        _studio_hub = EventHub()
        _studio_dispatch = _StudioDispatch()
        # Built here, explicitly, and threaded through to `serve_studio_api`
        # rather than left for `create_app()`'s own default -- its default
        # is a *private* store scoped to whichever app happens to build it,
        # which is fine for a test building its own app but means nothing
        # outside this function could ever reach it.
        #
        # Held in a LOCAL until serve_studio_api() has returned without
        # raising -- not assigned to the module global up front. serve()
        # calls create_app(), which does an eager `vault.instance_secret()`
        # check that raises synchronously on a bad TENKA_SECRET (see the
        # comment above on `_studio_hub`/subscribe timing for the identical
        # reasoning already applied there). Assigning the global before that
        # call, the same way this function used to, left `_studio_pair_store`
        # pointing at a real, live-looking `PairCodeStore` even when
        # `serve()` never returned a task at all -- `/studio pair` would
        # keep minting codes for a daemon that was never actually serving
        # `POST /v1/pair`, the identical "unredeemable code" failure the
        # post-stop cleanup fixes for the *other* end of the daemon's
        # lifetime. Ordering it this way makes the invariant ("the global is
        # set only when something is actually listening") hold by
        # construction -- there is no `except` branch left to forget to
        # undo it in, because the assignment that would need undoing simply
        # has not happened yet.
        _pair_store = PairCodeStore()
        _studio_task = serve_studio_api(
            build_studio_runtime(_studio_dispatch),
            _studio_vault,
            port=config.STUDIO_API_PORT,
            origins=[o.strip() for o in config.STUDIO_API_ORIGINS.split(",") if o.strip()],
            hub=_studio_hub,
            # Resolved here rather than inside the app, because choosing the
            # path means reading config and `io/api` may not. Resolved
            # *before* serve() so an unreadable bundle is one WARNING and a
            # UI-less daemon, never a daemon that failed to start: the API is
            # the product and the pages are a convenience.
            ui_bundle=_resolve_studio_ui(),
            pair_store=_pair_store,
        )
        _studio_pair_store = _pair_store
        # publish_status(), not publish() directly: it translates the
        # broadcaster's own snake_case event shape into the daemon's
        # camelCase wire frame (assistant/io/api/events.py) before it ever
        # reaches a socket -- status_broadcaster's shape stays its own,
        # unmodified, for its other consumer (the overlay's IPC writer).
        status.subscribe(_studio_hub.publish_status)
        return _studio_task
    except Exception as e:
        logger.warning(
            f"[API] Studio daemon failed to start (non-critical): {redact_secrets(str(e))}"
        )
        return None


async def _stop_studio_daemon(task: "asyncio.Task | None") -> None:
    """Cancel a running daemon task, or retrieve-and-log one that already
    crashed before shutdown got here (e.g. the port was already taken, so
    _run() raised inside the task instead of at the synchronous serve()
    call _start_studio_daemon() already guards).

    Retrieving a done task's exception marks it consumed, so asyncio's own
    "exception was never retrieved" logger never prints it later,
    unredacted, through a handler this module doesn't control.

    Also clears `_studio_pair_store`, unconditionally and first, regardless
    of which path below actually runs. A stale store is the exact "prints a
    code nobody can redeem" failure `/studio pair` exists to prevent -- just
    reached through a global that outlived the daemon instead of a fresh
    construction. `/studio pair` must see `None` the instant the daemon
    stops serving, not only after some particular shutdown path (no task,
    an already-crashed task, or a task actually cancelled here all count).
    """
    global _studio_pair_store
    _studio_pair_store = None
    if task is None:
        return
    if task.done():
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.warning(f"[API] Studio daemon exited early: {redact_secrets(str(exc))}")
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning(f"[API] Studio daemon did not shut down cleanly: {redact_secrets(str(e))}")


async def _drain_and_announce_notifications(bridge: UnityBridge):
    """
    Drain incoming message notifications from the messaging bridge,
    debounce by sender, and announce via TTS when the debounce window expires.

    Called every main loop cycle (~0.1s).
    """
    global _notification_debounce_buffer

    # Step 1: Drain raw notifications into debounce buffer
    raw = messaging_bridge.drain_notifications()
    for notif in raw:
        if notif.get("type") != "incoming_message":
            continue
        
        # For now, only handle private messages. Group support planned.
        if notif.get("chat_type") != "private":
            continue

        service = notif.get("service", "")
        if not service:
            continue
        sender = notif.get("sender", "")
        key = (service, sender)

        if key not in _notification_debounce_buffer:
            _notification_debounce_buffer[key] = {
                "sender_name": notif.get("sender_name", sender),
                "messages": [],
                "first_seen": _time.time(),
                "chat_type": notif.get("chat_type", "private"),
                "service": service,
            }

        _notification_debounce_buffer[key]["messages"].append({
            "text": notif.get("text", ""),
            "timestamp": notif.get("timestamp", ""),
        })
        # Update sender name in case cache improved since first message
        if notif.get("sender_name", "") and notif["sender_name"] != sender:
            _notification_debounce_buffer[key]["sender_name"] = notif["sender_name"]

    # Step 2: Evict stale entries that have been deferred too long
    now = _time.time()
    debounce_sec = getattr(config, "MESSAGING_NOTIFY_DEBOUNCE", 5.0)
    _evict_cutoff = now - debounce_sec * 20
    _notification_debounce_buffer = {
        k: v for k, v in _notification_debounce_buffer.items()
        if v["first_seen"] > _evict_cutoff
    }

    # Step 3: Check for batches whose debounce window has expired
    ready_keys = [
        k for k, v in _notification_debounce_buffer.items()
        if now - v["first_seen"] >= debounce_sec
    ]

    if not ready_keys:
        return

    # Step 4: Check if we should defer (active pending states or recording)
    from .pending import pending_registry
    has_pending = (
        pending_registry.any_active(exclude={"incoming_messages", "teaching_session"})
        or recording.is_active()
    )
    if has_pending:
        # Don't announce now — leave in buffer, will check next cycle
        return

    # Step 5: Announce each ready batch
    from . import actions as _actions
    for key in ready_keys:
        batch = _notification_debounce_buffer.pop(key)
        sender_name = batch["sender_name"]
        msg_count = len(batch["messages"])
        service_name = batch["service"].title()

        if msg_count == 1:
            announcement = f"You got a {service_name} message from {sender_name}."
        else:
            announcement = f"You got {msg_count} {service_name} messages from {sender_name}."

        logger.info(f"[NOTIFY] {announcement}")

        # Speak the announcement
        await tts.speak(announcement, bridge, emotion="neutral")

        # Set pending state so user can say "read it"
        msgs = _actions.pending_incoming_messages.payload or []
        msgs.append(batch)
        _actions.pending_incoming_messages.set(msgs)

    # Keep only the most recent 5 batches to prevent unbounded growth
    msgs = _actions.pending_incoming_messages.payload
    if msgs and len(msgs) > 5:
        _actions.pending_incoming_messages.set(msgs[-5:])


# ─── Implicit procedure management (no "procedure" keyword) ─────────────────

_IMPLICIT_PROC_EDIT_RE = re.compile(
    r"^(?:edit|modify|update|re-?teach|redo)\s+(?:the\s+)?(.+)$", re.I,
)
_IMPLICIT_PROC_DELETE_RE = re.compile(
    r"^(?:delete|remove|forget|forgot|drop)\s+(?:the\s+)?(.+)$", re.I,
)


def _match_implicit_proc_command(text: str):
    """
    Catch 'edit X' / 'delete X' when X matches a known procedure.
    Only fires when regex_router didn't match (no 'procedure' keyword).
    Zero LLM cost — just a regex + DB lookup.
    """
    from .intent import IntentResult

    for pattern, action in (
        (_IMPLICIT_PROC_EDIT_RE, "edit"),
        (_IMPLICIT_PROC_DELETE_RE, "delete"),
    ):
        m = pattern.match(text.strip())
        if not m:
            continue
        name = m.group(1).strip().rstrip(".!?")
        if not name or len(name) < 3:
            continue
        proc = procedures.find_by_name_or_trigger(name)
        if proc:
            logger.info(f"[PROC-MGMT] Implicit {action} → '{proc['trigger']}'")
            return IntentResult(
                intent="manage_procedure", response=text,
                params={"action": action, "name": name},
            )
    return None


# ─── Teach trigger patterns ──────────────────────────────────────────────────

_TEACH_PATTERNS = [
    re.compile(r"(?:let me |i want to |i'll )?teach you (?:how to |to )?(.+)", re.IGNORECASE),
    re.compile(r"(?:let me |i'll )?show you how to (.+)", re.IGNORECASE),
    re.compile(r"create (?:a )?procedure (?:for |to )?(.+)", re.IGNORECASE),
    re.compile(r"new procedure (?:for |to )?(.+)", re.IGNORECASE),
]


def _match_teach_trigger(text: str) -> str | None:
    """
    Return the name_seed if text matches a teach trigger, else None.
    Strips leading filler words (and, so, ok, hey, …) before matching so
    STT artifacts like "and let me teach you…" are handled correctly.
    """
    raw     = text.strip()
    cleaned = re.sub(r'^(?:and|so|okay|ok|hey|uh|um|well|now|alright)\s+',
                     '', raw, flags=re.IGNORECASE)
    for pat in _TEACH_PATTERNS:
        for candidate in (cleaned, raw):
            m = pat.fullmatch(candidate) or pat.match(candidate)
            if m:
                seed = m.group(1).strip().rstrip(".!?").strip()
                if seed and len(seed) >= 3:
                    return seed
    return None


def _match_batch_teach(text: str) -> tuple[str, str] | None:
    """
    Detect a multi-line batch teach input. The first line must match a teach
    trigger and at least one subsequent line must exist as a step.
    Returns (name_seed, body) or None.
    """
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None
    seed = _match_teach_trigger(lines[0])
    if not seed:
        return None
    body = "\n".join(lines[1:])
    return (seed, body)


# ─── Which turn sources are a person at this machine ─────────────────────────

_LOCAL_SOURCES: frozenset[str] = frozenset({"stt", "chat"})
"""Sources that mean "someone is physically at this keyboard or microphone".

Spelled as a literal allow-list, never as `!= "studio"`. Two things hang off
it and both must fail closed on a source nobody has added yet: whether TENKA
speaks the answer out loud in the operator's room, and whether she reopens the
microphone for a follow-up. `_grants_for_item` hands an `stt` item the full
local grant set, so "may this turn reopen the microphone" is the same question
as "may this turn mint full local privilege" -- and the answer for an unknown
source has to be no.

`"studio"` is deliberately absent even though desktop Studio runs on this
machine: it arrives over HTTP, and the slash-command branch already refuses it
on exactly that reasoning.
"""


# ─── Pending Handler Table ───────────────────────────────────────────────────
# The dispatcher in process_text_from_queue() walks this in order; the first
# handler that returns non-None owns the turn. Registering a pending state in
# actions/__init__.py is NOT enough — a handler missing from this table is
# simply never called, and its state silently expires. Module-level (not
# function-local) so tests can assert every exported handle_pending_* is here.
#
# Each entry: (handler_func, log_label, memory_intent, needs_bridge, capability)
#
# The fifth column is what the handler's *effect* costs, and the dispatch loop
# skips any row the turn's grants do not cover. It is a column rather than a
# lookup keyed off memory_intent for two reasons: six of the rows name a
# pending-only label that is not an intent at all, and memory_intent is a
# storage label — overloading it as a permission declaration would couple two
# unrelated things and make the next handler's capability an accident rather
# than a decision. Adding a row without one is a ValueError on the loop's
# unpack, and tests/test_6a5_predispatch_gate.py names it earlier than that.
#
# The reasoning per row, since none of it is obvious from the name:
#   DESTRUCTIVE / FILE     — confirm a delete, pick a search hit → FILES, the
#                            same capability `file_task` itself costs.
#   CAMERA / FACE          — the camera and the face database → SCREEN.
#   OAUTH / DEVICE_AUTH    — these paste provider client secrets and pair
#                            third-party accounts into the machine's own
#                            credential store. That is reconfiguring the
#                            machine, not using it → SYSTEM_CONTROL, the same
#                            reasoning `enroll_voice` carries.
#   MESSAGING (both)       — disambiguation *retries the original send*, so
#                            both rows end in messaging_bridge driving an app
#                            → EXECUTE.
#   INCOMING               — reads the owner's received messages back into the
#                            transcript → RECALL, the read capability. A
#                            device that holds RECALL can already read the
#                            transcript, so this is consistent, not a hole.
#   KNOWLEDGE              — approves storing a fact → CHAT_SEND, matching
#                            `store_memory`.
#   MONITOR / BACKUP       — `manage_monitor` / `manage_backup` → EXECUTE.

from .actions import (
    handle_pending_destructive, handle_pending_camera_settings,
    handle_pending_forget_face, handle_pending_file_search,
    handle_pending_oauth_setup, handle_pending_device_auth,
    handle_pending_messaging_disambig, handle_pending_messaging_send,
    handle_pending_incoming_message, handle_pending_knowledge_approval,
    handle_pending_monitor_disambig, handle_pending_backup_confirm_phrase,
    handle_pending_backup_oauth, handle_pending_backup_unlock_phrase,
    handle_pending_backup_restore_phrase,
)

_PENDING_HANDLERS = [
    (handle_pending_destructive,        "DESTRUCTIVE",  "file_task",          False, Capability.FILES),
    (handle_pending_camera_settings,    "CAMERA",       "camera_look",        False, Capability.SCREEN),
    (handle_pending_forget_face,        "FACE",         "forget_face",        False, Capability.SCREEN),
    (handle_pending_file_search,        "FILE",         "file_task",          False, Capability.FILES),
    (handle_pending_oauth_setup,        "OAUTH",        "oauth_setup",        True,  Capability.SYSTEM_CONTROL),
    (handle_pending_device_auth,        "DEVICE_AUTH",  "device_auth",        True,  Capability.SYSTEM_CONTROL),
    (handle_pending_messaging_disambig, "MESSAGING",    "messaging_disambig", True,  Capability.EXECUTE),
    (handle_pending_messaging_send,     "MESSAGING",    "messaging_send",     False, Capability.EXECUTE),
    (handle_pending_incoming_message,   "INCOMING",     "incoming_message",   False, Capability.RECALL),
    (handle_pending_knowledge_approval, "KNOWLEDGE",    "knowledge_approval", True,  Capability.CHAT_SEND),
    (handle_pending_monitor_disambig,   "MONITOR",      "manage_monitor",     False, Capability.EXECUTE),
    (handle_pending_backup_confirm_phrase, "BACKUP",    "manage_backup",      False, Capability.EXECUTE),
    (handle_pending_backup_oauth,       "BACKUP",       "manage_backup",      False, Capability.EXECUTE),
    (handle_pending_backup_unlock_phrase, "BACKUP",     "manage_backup",      False, Capability.EXECUTE),
    (handle_pending_backup_restore_phrase, "BACKUP",    "manage_backup",      False, Capability.EXECUTE),
]


# ─── Wake Word Listener (global reference) ───────────────────────────────────

_wake_listener: WakeWordListener | None = None

# ─── Session Continuity (set during startup in _run_async) ───────────────────

_session_resume_context: str = ""


# ─── Pipeline ────────────────────────────────────────────────────────────────


async def run_pipeline(bridge: UnityBridge, from_wake_word: bool = False):
    """
    Run the STT part of the pipeline:
      Mic → STT → Queue

    Args:
        from_wake_word: If True, this was triggered by wake word detection.
                        Affects speaker verification behavior.
    """
    # Pause wake word detection during the pipeline (avoid double triggers)
    if _wake_listener:
        _wake_listener.pause()

    try:
        # Tell Unity we're "thinking"
        await bridge.send_command("play_animation", name="thinking")

        # Step 1: STT — stop recording and transcribe
        audio = recorder.stop()

        # Parallel speaker verification + transcription
        # ECAPA runs in ~50-100ms, Whisper in ~300-500ms.
        # Running in parallel means verification is "free" — hidden by Whisper latency.
        if config.SPEAKER_VERIFY_ENABLED and speaker_verify.is_enrolled():
            import concurrent.futures
            _stt_start = _time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                sv_future = pool.submit(speaker_verify.verify, audio)
                stt_future = pool.submit(transcribe, audio)

                sv_result = sv_future.result()
                transcription = stt_future.result()
            _stt_ms = int((_time.monotonic() - _stt_start) * 1000)

            if not sv_result["is_owner"]:
                logger.info(
                    f"[SV] Rejected unknown speaker in pipeline "
                    f"(score={sv_result['score']:.3f}, "
                    f"threshold={sv_result['threshold']:.3f}, "
                    f"method={sv_result['method']})"
                )
                # Silent discard — regardless of source
                if _wake_listener:
                    _wake_listener.resume()
                return
        else:
            _stt_start = _time.monotonic()
            transcription = transcribe(audio)
            _stt_ms = int((_time.monotonic() - _stt_start) * 1000)

        if not transcription.strip():
            logger.warning("No transcription received")
            await tts.speak("I didn't catch that. Could you try again?", bridge)
            if _wake_listener:
                _wake_listener.resume()
            return

        # Instead of processing directly, push to queue
        _input_queue.put(("stt", transcription, _stt_ms))

    except Exception as e:
        logger.error(f"STT Pipeline error: {e}", exc_info=True)
        try:
            await tts.speak("Sorry, something went wrong with recording.", bridge)
        except Exception:
            pass
        if _wake_listener:
            _wake_listener.resume()


# ─── Pipeline Processing ────────────────────────────────────────────────────────


def _publish_turn_status(source: str, phase: StatusPhase) -> None:
    """Bracket a remote-sourced turn with a lifecycle status transition.

    Studio settles a live turn on a *quiet window of status frames*: every
    frame re-arms a timer (`hooks/useEventStream.ts`), a terminal phase
    starts the window, and `settleLiveTurn()` then re-reads the transcript
    and renders the reply. Nothing on that wire names a turn or a
    conversation, so the frames themselves are the only end-of-turn signal a
    remote pane has -- `POST /v1/chat` is 202 Accepted with no body, and this
    function's return value never leaves the process.

    Until now the only producer on an ordinary chat turn's path was
    `tts.speak()`, and only incidentally: `SPEAKING` on entry, `IDLE` on
    exit (io/audio/tts.py). So every branch of this pipeline that *answers
    without speaking* left a Studio caller's pane pending forever -- no
    reply rendered, composer stuck on "stop" -- which is exactly what
    exempting the "studio" source from speech turned into a visible bug.
    Bracketing the turn here, rather than patching whichever branch was
    reported, is what makes the whole class impossible: the pair fires
    however the turn exits, including through routes nobody has written yet.

    Both ends are needed; one terminal frame is not enough. `StatusBroadcaster
    .set()` drops an event identical to the previous one, and a bare `IDLE`
    with default fields is precisely the resting state every completed turn
    already leaves behind -- so a lone closing frame would be silently
    swallowed in the common case (a refusal typed right after any normal
    turn). The opening `THINKING` both tells the pane she picked the turn up
    and guarantees the closing `IDLE` is a real transition rather than a
    duplicate.

    `detail` is left empty deliberately. The field is RECALL-class and
    blanked per-socket (`io/api/events.py`), so anything put there is
    invisible to an observe-only device -- a signal half the clients cannot
    see is not the one to hang end-of-turn on. And a refusal sentence is
    content, not status: it already reaches the pane as a transcript
    message, so echoing it here would render it twice for a device that may
    read it and as an empty pill for one that may not. `stt.py` publishes
    `LISTENING, detail=""` for the same reason -- a pure lifecycle
    transition describes her, not what was said.

    Local sources are deliberately untouched. The desktop overlay reads the
    same bus but has no settle model -- it renders whatever pill the running
    handler publishes -- so a second producer on every voice turn would buy
    nothing and risk a spurious "Done" flash on transitions that did no work.
    """
    if source != "studio":
        return
    try:
        status.set(phase)
    except Exception:
        # The contract every other producer honours (see stt.py, tts.py):
        # the status bus must never take down the turn it describes.
        pass


async def process_text_from_queue(source: str, transcription: str, bridge: UnityBridge,
                                  stt_ms: int | None = None,
                                  grants: "frozenset[Capability] | None" = None):
    """
    Run Intent → Policy → Action/LLM → TTS → Unity animations

    `grants` is what the caller driving this turn is allowed to ask for.
    `None` means "nobody said", and `actions.execute()` refuses everything in
    that state -- it is not a synonym for the local full set. Callers that
    know the answer state it; `_grants_for_item` is the one that decides for
    a queued item.
    """
    global _turn_counter
    from . import session as session_mod
    _tracker = _telemetry.TurnTracker(
        session_id=session_mod.get_current_session_id(),
        input_modality="voice" if source == "stt" else "text",
        transcript=transcription,
    )
    _tracker.latency_stt_ms = stt_ms
    _tracker_token = _telemetry.set_current_tracker(_tracker)
    # Same token pattern as the tracker above, and reset in the same
    # `finally`. Every intent this turn dispatches -- including the ones a
    # planner step re-enters with -- is checked against this set.
    _t0_intent = _time.monotonic()

    # Pause wake word detection during execution
    if _wake_listener:
        _wake_listener.pause()

    # Set LAST, immediately before the `try` whose `finally` resets it, and
    # after `_wake_listener.pause()` rather than before. An adversarial
    # review found this three statements higher up: a raise anywhere in that
    # window skipped the reset entirely and left the grant set installed in
    # the queue consumer's context after the turn ended. The documented
    # fail-closed property inverts there -- whatever ran next inherited the
    # last turn's grants instead of none. Nothing may be added between this
    # line and the `try`.
    _grants_token = _actions_module.set_grants(
        frozenset() if grants is None else grants)

    try:
        # ─── The pre-dispatch gate ───────────────────────────────────────
        # `actions.execute()` checks a turn's grants immediately before it
        # resolves a handler, and that is the only site in the tree that
        # resolves one. It is still not the only site that produces an
        # *effect*: everything between here and `execute_action(...)` below
        # can teach a procedure, replay one, flip speaker verification, kill
        # the process, or drive a pending confirmation, and every one of
        # those used to `return` before the gate was ever reached.
        #
        # This closure is what a branch calls to say "what I am about to do
        # costs `required`". It answers with `actions.capability_refusal` --
        # the same predicate `execute()` uses, reading the same contextvar --
        # so there is one source of truth about this turn's grants rather
        # than one per door. It returns True when the branch must not run,
        # so a call site reads:
        #
        #     if await _gate(Capability.EXECUTE, "shutdown", "shutdown"):
        #         return
        #
        # `tests/test_6a5_predispatch_gate.py` walks this function's AST and
        # fails if any branch that returns before dispatch calls neither this
        # nor `capability_refusal` directly. That test, not this comment, is
        # what stops the next branch reopening the hole.
        #
        # Not every branch refuses. One that the *caller* asked for refuses
        # and says so; one that answers state the caller did not arm (the
        # teaching session, the pending chain) skips silently instead, by
        # consulting `capability_refusal` in its own condition -- refusing
        # there would both hijack a reply that should get an ordinary turn
        # and disclose that a confirmation is waiting.
        async def _respond(text: str, mem_intent: str,
                           emotion: str = "neutral") -> None:
            """Deliver a pre-dispatch branch's answer to whoever asked for it.

            Three defects in this region had one shape, and this is their one
            fix.

            *It always records the turn, under the session id.* Studio settles
            a turn by re-reading the transcript
            (`LiveChatRuntime.conversation()` -> `memory.get_recent`), never
            from this function's return value -- `POST /v1/chat` is 202 with no
            body. Four branches here saved under `date.today().isoformat()`
            instead of the session id, which reads as "saved" and renders as
            "the turn vanished"; three saved nothing at all.

            *It speaks only to a local source.* A remote device that can make
            the local speaker talk on demand has a standing way to interrupt
            the owner's room from off the machine.

            *It reopens the microphone only for a local source.* This is the
            serious one. `_finish_turn` runs `_follow_up_listen()`, which runs
            `record_until_silence()` and re-queues the result as
            `("stt", ...)`, and `_grants_for_item` hands an `stt` item the
            FULL local grant set. A remote message that merely matched the
            teach-trigger regex therefore opened the microphone and turned the
            next thing said in the room into a fully privileged turn. Remote
            hot mic and remote-to-local escalation in one move. `_finish_turn`
            refuses non-local sources itself as well; this is the near half of
            that pair.
            """
            memory.save_turn(transcription, mem_intent, text,
                             session_mod.get_current_session_id())
            if source in _LOCAL_SOURCES:
                await tts.speak(text, bridge, emotion=emotion)
                await _finish_turn(bridge, source)

        async def _gate(required: Capability, branch: str,
                        mem_intent: str) -> bool:
            refusal = _actions_module.capability_refusal(required)
            if refusal is None:
                return False
            logger.info(
                f"[GATE] Refused pre-dispatch branch {branch!r}: "
                f"needs {required.value}"
            )
            await _respond(refusal, mem_intent, "calm")
            _tracker.intent_detected = branch
            _tracker.action_outcome = "refused"
            _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
            return True

        # `!r`, not hand-written quotes. A newline in the text used to end the
        # log line and start a second one that an operator grepping debug.log
        # after an incident would read as a real entry -- attacker-authored
        # text choosing what the audit trail says. repr() escapes the newline
        # to `\n` and keeps the whole thing on one quoted line. redact_secrets
        # is the other half and does not cover this: it removes secrets, not
        # line breaks.
        if source == "stt":
            logger.info(f'Transcription (STT): {redact_secrets(transcription)!r}')
        else:
            logger.info(f'Transcription (Chat): {redact_secrets(transcription)!r}')

        # The turn has begun. See _publish_turn_status: this is a no-op for
        # every local source, and for "studio" it is the frame that replaces
        # what tts.speak() used to provide by accident.
        _publish_turn_status(source, StatusPhase.THINKING)

        await bridge.send_command("show_subtitle", text=f"You: {transcription}")

        # ─── Slash commands (zero-LLM runtime config) ────────────────────
        # Chat input like "/set followup_timer 7" is intercepted here before
        # any teaching / shortcut / intent processing. Voice rarely produces
        # leading "/", so this is effectively console/voice-only in practice
        # -- deliberately, not by accident. `source == "studio"` is the one
        # exception that must never execute: POST /v1/chat requires only
        # CHAT_SEND, and _StudioDispatch.submit() -> this function applies no
        # further check per source. CHAT_SEND is in every /studio pair
        # default grant, so without this refusal a device holding nothing
        # but CHAT_SEND could type "/studio pair evilphone" (or
        # "/studio revoke all confirm", or "/set", or "/reset" -- the whole
        # slash surface, not just pairing) over chat and reach commands that
        # were never gated by the capability system at all. That is the
        # sideways escalation Task 10 closed on POST /v1/pair/code, reopened
        # through a different door. Slash commands stay a console/voice
        # affordance; Studio's own typed routes (Settings, Devices) are
        # where a remote caller's grants actually apply.
        # NOTE: no follow-up listen — config commands aren't conversation.
        # The `finally` block at the bottom of this function resumes the
        # wake listener (and, for "studio", clears _StudioDispatch's busy
        # flag) unconditionally, so bailing out here without _finish_turn()
        # is safe for every source, including studio: calling it would open
        # the local microphone for a follow-up listen off the back of a
        # remote HTTP request, which is a worse problem than the one below.
        if slash_commands.is_slash_command(transcription):
            if source == "studio":
                response = ("Slash commands aren't available through "
                            "Studio's chat -- use its Settings and Devices "
                            "pages for this.")
                outcome = "refused"
                # Studio settles a turn by re-reading the conversation
                # transcript (LiveChatRuntime.conversation() ->
                # memory.get_recent(conversation_id)), not from this
                # function's return value -- POST /v1/chat is 202 Accepted
                # with no body. Without this, the pane kept showing
                # whichever turn was last recorded, paired against a
                # message it had nothing to do with. conversation_id is
                # session_mod.get_current_session_id(), the same id
                # _StudioDispatch.submit() handed back for this turn, so
                # this is the record that id's re-read actually needs --
                # matching how the pending-handler chain below saves a
                # turn, not a new mechanism.
                memory.save_turn(transcription, "slash_command", response,
                                 session_mod.get_current_session_id())
            else:
                # Depth behind the source check above, not a replacement for
                # it. The source check is the stricter of the two and stays
                # first: it refuses Studio even when the device holds
                # everything, because the slash surface reaches pairing and
                # revocation. This second check is what covers a source that
                # does not exist yet -- the slash surface writes runtime
                # config, which is what PATCH /v1/settings charges
                # SYSTEM_CONTROL for, so it charges the same. Voice and
                # console hold LOCAL_GRANTS, so it never fires today.
                if await _gate(Capability.SYSTEM_CONTROL, "slash_command",
                               "slash_command"):
                    return
                response = slash_commands.handle(transcription)
                outcome = "success"
            if source == "chat":
                print(response)
            elif source != "studio":
                # Speak a short confirmation only (full help text would be a
                # wall of speech). Truncate and keep the first line.
                #
                # Studio is excluded here for the same reason "chat" is:
                # both are text-native channels with their own rendered
                # surface (a console print / Studio's chat pane) rather than
                # a room TENKA talks in. A device holding nothing but
                # CHAT_SEND can already reach this refusal on every failed
                # attempt -- if it also made the local speaker narrate,
                # that device would have a standing way to make TENKA talk
                # in the owner's room on demand. The refusal is still
                # fully visible where it was asked (the Studio pane, via
                # the save above); an owner who wants an audible heads-up
                # on refusals has that logged already through the
                # transcription log line above, without turning every
                # remote probe into a spoken interruption.
                spoken = response.split("\n", 1)[0][:200]
                await tts.speak(spoken, bridge, emotion="neutral")
            _tracker.intent_detected = "slash_command"
            _tracker.intent_source = "regex"
            _tracker.action_outcome = outcome
            _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
            return

        # Sanitize Windows paths in input — raw backslashes break JSON parsing
        # Replace single backslashes with forward slashes for LLM processing
        # The original transcription is preserved for display/memory purposes
        intent_input = transcription.replace("\\", "/")

        # ─── Teaching session (before shortcuts so they can't fire mid-session) ────
        # Skipped, not refused, for a caller without EXECUTE. `teaching_session`
        # is process-global: a session armed at the keyboard would otherwise
        # eat the next message from any paired device and write its words into
        # the operator's procedure. Skipping sends that message through the
        # ordinary pipeline instead and leaves the session waiting for the
        # person who started it.
        if _actions_module.teaching_session.active and \
                _actions_module.capability_refusal(Capability.EXECUTE) is None:
            from .actions import handle_pending_teaching
            _teach_resp = await handle_pending_teaching(transcription)
            if _teach_resp is not None:
                _teach_emo, _ = llm.parse_emotion_tag(_teach_resp)
                await _respond(_teach_resp, "teach", _teach_emo or "happy")
                _tracker.intent_detected = "teaching"
                _tracker.intent_source = "regex"
                _tracker.action_outcome = "success"
                _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
                return

        # Batch teaching (multi-line paste: first line = teach trigger, rest = steps)
        _batch = _match_batch_teach(intent_input)
        if _batch:
            # EXECUTE, the same capability `manage_procedure` costs. A taught
            # procedure is a stored keystroke/click/app-launch program, and
            # `open cmd` / `type <anything>` / `press enter` all parse. That
            # installing one is inert until it is triggered is not an argument
            # for a weaker capability: the device that installs it is the
            # device that can trigger it, so the pair is one privilege, and
            # the installer is the half that is hardest to notice after the
            # fact. Same reasoning the spec gives for the four `manage_*`
            # intents -- gating the installed thing and not the installer
            # would be theatre.
            if await _gate(Capability.EXECUTE, "batch_teaching",
                           "batch_teaching"):
                return
            from .actions import start_batch_teaching
            _batch_resp = start_batch_teaching(_batch[0], _batch[1])
            await _respond(_batch_resp, "batch_teaching", "happy")
            _tracker.intent_detected = "batch_teaching"
            _tracker.intent_source = "regex"
            _tracker.action_outcome = "success"
            _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
            return

        # Teaching trigger detection (zero LLM cost, enters collecting state)
        _teach_match = _match_teach_trigger(intent_input)
        if _teach_match:
            # Same capability as the batch form above, and it must be: this is
            # the same install reached one line at a time. Leaving it cheaper
            # would just mean the multi-turn route is the one an attacker uses.
            if await _gate(Capability.EXECUTE, "teaching_trigger",
                           "teaching_trigger"):
                return
            from .actions import start_teaching_session
            _opening = start_teaching_session(_teach_match)
            await _respond(_opening, "teaching_trigger", "happy")
            _tracker.intent_detected = "teaching_trigger"
            _tracker.intent_source = "regex"
            _tracker.action_outcome = "success"
            _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
            return

        # ─── Procedure management commands (before execution) ─────────────
        _proc_cmd = regex_router.match_procedure_command(intent_input)

        if not _proc_cmd:
            _proc_cmd = _match_implicit_proc_command(intent_input)

        # ─── Procedure execution (before shortcuts, before intent) ────────
        if not _proc_cmd:
            _proc_match = procedures.match_trigger(intent_input)
            if _proc_match:
                # EXECUTE. `run_procedure` drives `automation.native` and
                # `pyautogui` directly and never re-enters `actions.execute()`,
                # so nothing downstream of this call checks anything. It now
                # checks itself as a backstop too -- this is the boundary, that
                # is the depth.
                if await _gate(Capability.EXECUTE, "procedure", "procedure"):
                    return
                from . import procedure_executor
                logger.info(f"[PROC] Executing '{_proc_match['name']}' ({len(_proc_match['steps'])} steps)")
                _proc_result = await procedure_executor.run_procedure(_proc_match, transcription)
                _proc_spoken = await llm.chat(
                    f"Summarize in 1-2 short sentences for voice (friendly, no bullet points): {_proc_result}",
                    task_type="synthesis",
                )
                _proc_emo, _proc_clean = llm.parse_emotion_tag(_proc_spoken)
                await _respond(_proc_clean, "procedure", _proc_emo or "happy")
                _tracker.intent_detected = "procedure"
                _tracker.intent_source = "procedure"
                _tracker.action_dispatched = _proc_match["name"]
                _tracker.action_outcome = "success"
                _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
                return

        # Step 1b: Shortcut check — skip intent classifier if shortcut matches
        _shortcut_match = shortcuts.match_shortcut(transcription) if not _proc_cmd else None

        # Listen mode toggle (pre-intent, zero LLM cost)
        if config.SPEAKER_VERIFY_ENABLED and not _shortcut_match and not _proc_cmd:
            _sv_lowered = transcription.strip().lower()

            _LISTEN_ALL = [
                "listen to everyone", "let others talk to you",
                "open mic mode", "let anyone talk to you",
            ]
            _LISTEN_OWNER = [
                "only listen to me", "stop listening to everyone",
                "close mic mode", "only respond to me",
            ]

            if any(p in _sv_lowered for p in _LISTEN_ALL):
                # SYSTEM_CONTROL, because `PATCH /v1/settings` charges
                # SYSTEM_CONTROL for the very same runtime-config key. Two
                # doors onto one switch had two different prices, and this was
                # the free one. It is also the worst switch to leave free: with
                # verification off, anyone audible near the machine drives a
                # *voice* turn, and voice carries LOCAL_GRANTS -- so a remote
                # device holding only CHAT_SEND could convert itself into full
                # local privilege via anything that can play sound in the room.
                if await _gate(Capability.SYSTEM_CONTROL, "speaker_verify",
                               "speaker_verify"):
                    return
                speaker_verify.set_listen_to_everyone(True)
                resp = (
                    "[sarcastic] Fine, I'll listen to whoever. "
                    "Don't blame me if some random starts bossing me around."
                )
                parsed_emo, clean = llm.parse_emotion_tag(resp)
                await _respond(clean, "speaker_verify", parsed_emo or "sarcastic")
                _tracker.intent_detected = "speaker_verify"
                _tracker.intent_source = "regex"
                _tracker.action_outcome = "success"
                _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
                return

            if any(p in _sv_lowered for p in _LISTEN_OWNER):
                # The same switch, gated the same way. This direction tightens
                # rather than loosens, so it is not an escalation -- but it
                # still lets a remote caller overwrite a setting the operator
                # chose deliberately, and one switch with two prices depending
                # on which way you push it is how the hole above got missed.
                if await _gate(Capability.SYSTEM_CONTROL, "speaker_verify",
                               "speaker_verify"):
                    return
                speaker_verify.set_listen_to_everyone(False)
                resp = (
                    "[happy] Back to just you and me. "
                    "The way it should be... n-not that I prefer it or anything!"
                )
                parsed_emo, clean = llm.parse_emotion_tag(resp)
                await _respond(clean, "speaker_verify", parsed_emo or "happy")
                _tracker.intent_detected = "speaker_verify"
                _tracker.intent_source = "regex"
                _tracker.action_outcome = "success"
                _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
                return

        _regex_result = None
        if _proc_cmd:
            logger.info(f"[PROC-MGMT] {_proc_cmd.params.get('action')}")
            intent_result = _proc_cmd
        elif _shortcut_match:
            logger.info(f"[SHORTCUT] Matched '{_shortcut_match['trigger']}' → {_shortcut_match['intent']}")
            from .intent import IntentResult
            intent_result = IntentResult(
                intent=_shortcut_match["intent"],
                response=transcription,
                params=_shortcut_match["params"],
            )
        else:
            # Step 1c: Regex pre-router — zero API cost for common commands
            _regex_result = regex_router.pre_route(intent_input)
            if _regex_result:
                logger.info(f"[REGEX] Matched → {_regex_result.intent}")
                intent_result = _regex_result
            else:
                # Step 2: Intent Detection — classify the text
                # resolve pronouns and get topic hint
                _tracker_inst = _get_topic_tracker()
                _resolved_input = _tracker_inst.resolve_query(intent_input)
                _topic_hint = _tracker_inst.get_topic_hint()

                # detect active scope from system state
                from .intent_scopes import detect_scope
                _scope_name, _active_intents = detect_scope(_turn_counter)

                intent_result = await detect_intent(
                    _resolved_input,
                    scope=_scope_name,
                    active_intents=_active_intents,
                    topic_hint=_topic_hint,
                )
        logger.info(f"Intent: {intent_result.intent}")

        _tracker.latency_intent_ms = int((_time.monotonic() - _t0_intent) * 1000)
        _tracker.intent_detected = intent_result.intent
        if _proc_cmd:
            _tracker.intent_source = "procedure"
        elif _shortcut_match:
            _tracker.intent_source = "shortcut"
        elif _regex_result:
            _tracker.intent_source = "regex"
        else:
            _tracker.intent_source = "llm"

        _telemetry.check_correction(_tracker)

        # ── Shutdown intent — graceful exit ──────────────────────────
        if intent_result.intent == "shutdown":
            # `core/intent_capabilities.py` already puts `shutdown` behind
            # EXECUTE -- but the intent is handled here, inline, and returns
            # before `execute_action()` is ever called, so that row was dead
            # code and any CHAT_SEND device had a remote off switch for a
            # security product. The row is now enforced where the intent is
            # actually served.
            if await _gate(Capability.EXECUTE, "shutdown", "shutdown"):
                return
            _tracker.action_dispatched = "shutdown"
            _tracker.action_outcome = "success"
            await _respond("Shutting down. See you later!", "shutdown", "happy")
            _shutdown_event.set()
            return

        # Track session turn
        session_mod.record_turn(intent_result.intent)

        # push entities from this turn to the topic tracker
        _turn_counter += 1
        try:
            _get_topic_tracker().push_turn(transcription, _turn_counter)
        except Exception as e:
            logger.debug(f"[TOPIC] Push failed (non-critical): {e}")

        # ── Pending handler chain — generic dispatcher ────────────────
        # Table lives at module level (_PENDING_HANDLERS, near the top) so
        # registration is testable; see tests/test_pending_dispatch_wiring.py.
        # Plan resume is handled ONCE at the end — no per-handler changes.
        pending_handled = False
        pending_response = None

        for handler, label, mem_intent, needs_bridge, required in _PENDING_HANDLERS:
            # Skipped, not refused. Pending state is process-global, so the
            # device that ANSWERS a confirmation need not be the one that
            # ASKED: the operator says "delete my downloads" at the keyboard,
            # TENKA arms `pending_destructive`, and any paired device that
            # sends "yes" first drives the deletion. Charging the answer what
            # the effect costs closes the cases where the answering device
            # holds less than the handler needs.
            #
            # It does NOT close the case where it holds exactly enough -- a
            # tunnel device holds FILES, so it can still answer a locally
            # armed file confirmation. That residual is a missing owner on
            # PendingState, not a missing capability, and forcing a stronger
            # capability here to paper over it would break ordinary local
            # confirmations for no security gain. See the xfail in
            # tests/test_6a5_predispatch_gate.py.
            #
            # Skipping rather than refusing is deliberate twice over: the
            # caller did not ask for this handler, so its text deserves an
            # ordinary turn; and a refusal would disclose to a device that
            # cannot use it that a confirmation is waiting.
            if _actions_module.capability_refusal(required) is not None:
                logger.info(f"[{label}] Skipped: caller lacks {required.value}")
                continue
            if needs_bridge:
                resp = await handler(transcription, bridge)
            else:
                resp = await handler(transcription)
            if resp is not None:
                logger.info(f"[{label}] Handled pending state")
                parsed_emotion, _ = llm.parse_emotion_tag(resp)
                if parsed_emotion is None:
                    # Pending responses are hardcoded strings — infer emotion cheaply
                    low = resp.lower()
                    if any(w in low for w in ("sorry", "error", "wrong", "didn't work", "failed")):
                        parsed_emotion = "worried"
                    elif any(w in low for w in ("great", "all set", "got it", "done", "!")):
                        parsed_emotion = "happy"
                    else:
                        parsed_emotion = "neutral"
                if source in _LOCAL_SOURCES:
                    await tts.speak(resp, bridge, emotion=parsed_emotion)

                # A pending handler (e.g. backup restore) can request a
                # graceful exit — it already closed the live DB connection
                # before this returned (see close_for_restore() in
                # orchestrator.run_restore()), and deliberately does not
                # reopen it: other modules cache their own repo/Database
                # reference at startup (memory.py's _repo, etc.), so a
                # fresh storage/db.py connection wouldn't reach them
                # anyway — only a real process restart rebuilds those
                # correctly. So from here on this turn must touch the DB
                # exactly as little as the "shutdown" intent's own branch
                # above does: no memory.save_turn, no planner-resume
                # (which can itself write to the DB) — straight to the
                # same finish + exit sequence.
                from .core import shutdown_signal
                if shutdown_signal.is_requested():
                    logger.info("[SHUTDOWN] Handler requested graceful exit")
                    _tracker.action_dispatched = f"pending_{label.lower()}"
                    _tracker.action_outcome = "success"
                    await _finish_turn(bridge, source)
                    _shutdown_event.set()
                    return

                memory.save_turn(
                    transcription, mem_intent, resp,
                    session_mod.get_current_session_id(),
                )
                pending_handled = True
                pending_response = resp
                break

        if pending_handled:
            # CAPABILITY-EXEMPT: the pending chain gated each handler before awaiting it
            # This is the loop's epilogue, not a door of its own -- it can only
            # be reached by a handler the loop already charged for. The plan
            # resume below re-enters `actions.execute()`, which charges again.
            _tracker.action_dispatched = f"pending_{label.lower()}"
            _tracker.action_outcome = "success"
            # If a planner plan is suspended, resume it now
            from .actions.planner import planner as _planner_module
            if _planner_module.has_suspended_plan():
                logger.info("[PLANNER] Resuming suspended plan after interaction")
                resume_result = await _planner_module.resume_plan(pending_response)
                if resume_result:
                    parsed_emotion, clean_result = llm.parse_emotion_tag(resume_result)
                    if source in _LOCAL_SOURCES:
                        await tts.speak(
                            clean_result, bridge,
                            emotion=parsed_emotion or "neutral"
                        )
                    memory.save_turn(
                        "[plan resumed]", "planner", clean_result,
                        session_mod.get_current_session_id(),
                    )

            await _finish_turn(bridge, source)
            return

        # ── Recording mode guard ────────────────────────────────────────────
        if recording.is_active() and intent_result.intent not in (
            "stop_recording", "get_recording", "summarize_recording"
        ):
            # CAPABILITY-EXEMPT: declining to act is not an effect
            # The branch drops the input and returns. There is nothing here a
            # caller could want that it does not already have by staying quiet.
            logger.info(
                f"[RECORDING] Ignoring pipeline input during active session: {transcription}"
            )
            _tracker.action_outcome = "skipped"
            return

        # Step 3: Policy — check if the intent is allowed
        policy = evaluate_policy(intent_result)

        if not policy.allowed:
            # CAPABILITY-EXEMPT: this branch is itself a refusal
            # Nothing runs on this path, so there is no effect to charge for.
            # Note what this branch is NOT: a security control the gate may
            # lean on. `evaluate_policy` judges `intent_result.params["goal"]`,
            # and the goal-override below overwrites that same field with the
            # raw transcription for six intents afterwards -- so the string the
            # policy approved is not always the string that runs. The
            # capability gate is deliberately independent of it.
            logger.warning(f"Policy DENIED: {policy.reason}")
            await bridge.send_command("set_expression", value="worried")
            if source == "studio":
                # Studio settles a turn by re-reading the transcript
                # (LiveChatRuntime.conversation() -> memory.get_recent),
                # never from this function's return value -- POST /v1/chat is
                # 202 Accepted with no body. This branch returned without
                # recording anything, so a policy refusal left the pane
                # showing the user's own message and no answer at all: the
                # turn looked lost rather than refused. Same fix, same
                # reason, as the slash-command refusal above.
                memory.save_turn(transcription, intent_result.intent,
                                 policy.safe_response,
                                 session_mod.get_current_session_id())
            else:
                # Not spoken for "studio", for the reason the slash-command
                # branch spells out: a remote device that can make the local
                # speaker talk on demand is a standing way to interrupt the
                # owner's room from off the machine. The refusal is fully
                # visible where it was asked, via the save above.
                await tts.speak(policy.safe_response, bridge, emotion="calm")
            await bridge.send_command("set_expression", value="neutral")
            _tracker.action_outcome = "skipped"
            return

        # ── Clear suspended plan if user moved on ─────────────────────
        # If we reached here, no pending handler claimed the input.
        # Any suspended plan is stale — the user has moved on.
        from .actions.planner import planner as _planner_module
        if _planner_module.has_suspended_plan():
            logger.info("[PLANNER] User moved on — clearing suspended plan")
            _planner_module.clear_suspended_plan()

        # ── Planner pre-check on cleanest available goal ─────────────────
        # The 8b intent classifier truncates multi-step goals (e.g.
        # "weather and play music" → params.goal = "weather"). But the raw
        # transcription can be polluted by mic feedback or token-glue (e.g.
        # whisper rendering "5Open" without a space when the previous TTS
        # tail bled into the next utterance). Try the LLM-extracted goal
        # first — it survives the latter case — then fall back to the raw
        # transcription which catches the truncation case.
        _extracted_goal = (
            (intent_result.params.get("goal") if intent_result.params else "") or ""
        ).strip()
        _planner_input: str | None = None
        if intent_result.intent not in ("manage_shortcut", "manage_procedure", "manage_schedule", "manage_monitor"):
            for candidate in (_extracted_goal, transcription):
                if candidate and _planner_module.needs_planning(candidate):
                    _planner_input = candidate
                    break
        if _planner_input is not None:
            logger.info("[PLANNER] Multi-step goal detected → overriding intent")
            intent_result.intent = "planner"
            intent_result.params = {"goal": _planner_input}

        # Override the goal param for goal-based intents. Default to the
        # LLM-extracted goal when it looks substantive — that survives
        # transcript pollution like "café résumé 5Open notepad" where the
        # raw transcription leaked the previous TTS tail. Fall back to the
        # raw transcription only when extraction is missing or suspiciously
        # short, which historically happened with the 8b intent classifier
        # echoing prompt instructions instead of the user's words.
        if intent_result.intent in ("planner", "code_executor",
                            "file_task", "set_reminder", "cancel_reminder",
                            "computer_task") and not _shortcut_match:
            extracted = (intent_result.params.get("goal") or "").strip() if intent_result.params else ""
            if extracted and len(extracted) >= 3:
                intent_result.params["goal"] = extracted
            else:
                intent_result.params["goal"] = transcription

        # Step 4: Generate response
        _use_streaming = False
        _stream = None
        if intent_result.intent in ("small_talk", "unknown"):
            # Check if we're waiting for a yes/no on recording summary
            pending = getattr(recording, "_pending_summary", None)
            if pending is not None:
                lowered = transcription.strip().lower()
                if any(w in lowered for w in ("yes", "yeah", "sure", "yep", "please", "go ahead")):
                    recording._pending_summary = None
                    chunks = pending["chunks"]
                    full_text = "\n".join(c["transcript"] for c in chunks)
                    summary_prompt = (
                        f"The following is a transcript of a voice recording session "
                        f"({len(chunks)} chunks, {pending['duration_seconds']}s total). "
                        f"Write a concise bullet-point summary of the key points:\n\n{full_text}"
                    )
                    response_text = await llm.chat(summary_prompt, task_type="default")
                elif any(w in lowered for w in ("no", "nope", "skip", "don't", "dont", "nah")):
                    recording._pending_summary = None
                    response_text = "Alright, summary skipped. Your session is saved and you can ask me about it anytime."
                else:
                    response_text = await llm.chat(transcription)
            else:
                # Build native multi-turn messages + optional compression
                messages, compressed_summary = await _build_conversation_messages()
                facts_context = _build_facts_context()

                context_parts = []
                # query-time context injection (prepended so the response
                # model sees graph-resolved entities/facts ahead of flat facts).
                try:
                    kg_block = knowledge_graph.build_kg_context(transcription)
                except Exception as e:
                    logger.debug(f"[KG] build_kg_context failed (non-critical): {e}")
                    kg_block = None
                if kg_block:
                    context_parts.append(kg_block)
                if _session_resume_context:
                    context_parts.append(_session_resume_context)
                if facts_context:
                    context_parts.append(facts_context)
                if compressed_summary:
                    logger.debug(f"[CC] Injecting compressed summary into system prompt")
                    context_parts.append(
                        f"EARLIER CONVERSATION SUMMARY:\n{compressed_summary}"
                    )

                if context_parts or messages:
                    from .llm.prompts import build_personality_prompt
                    memory_context = "\n\n".join(context_parts)
                    system_prompt_with_context = (
                        f"{build_personality_prompt()}\n\n"
                        f"{memory_context}\n\n"
                        f"Use the above context to give personalized, consistent responses. "
                        f"If the user's name is known, use it naturally. "
                        f"Do not repeat back all the facts — just use them naturally."
                    )
                    _name_count = system_prompt_with_context.count(config.ASSISTANT_NAME_DISPLAY)
                    logger.debug(f"[CC] System prompt: {_name_count}x name mentions, {len(messages)} messages")
                    _stream = llm.stream_for_small_talk(
                        transcription,
                        system_prompt=system_prompt_with_context,
                        messages=messages if messages else None,
                    )
                else:
                    _stream = llm.stream_for_small_talk(transcription)

                _use_streaming = True
                _tracker.action_dispatched = intent_result.intent
                _tracker.action_outcome = "success"
        else:
            # Tool execution — run the matched handler (async for computer agent)
            _t0_action = _time.monotonic()
            response_text = await execute_action(
                intent_result.intent,
                intent_result.params,
                llm_response=transcription,
                bridge=bridge,
            )
            _tracker.latency_action_ms = int((_time.monotonic() - _t0_action) * 1000)
            _tracker.action_dispatched = intent_result.intent
            # Don't clobber a handler-reported failure (mark_action_failure)
            if _tracker.action_outcome != "failure":
                _tracker.action_outcome = "success"

        if not _use_streaming:
            try:
                from .personalities import get_active_loader
                if get_active_loader().get_feature_flags().get("sycophancy_filter"):
                    from .personalities.sycophancy import strip_sycophantic_opener
                    response_text = strip_sycophantic_opener(response_text)
            except Exception:
                pass

        if not _use_streaming:
            logger.info(f'Response: "{response_text}"')

        # Step 5: Save conversation turn to memory
        # (Deferred until after Step 6 for streaming — text isn't available yet)
        def _save_turn(text: str):
            _, clean_response = llm.parse_emotion_tag(text)
            from . import session as session_mod
            session_id = session_mod.get_current_session_id()
            conv_row_id: int | None = None
            try:
                conv_row_id = memory.save_turn(
                    user_input=transcription,
                    intent=intent_result.intent,
                    response=clean_response,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to save conversation turn: {e}")
            # knowledge-graph H: link every entity/fact/relationship extracted from this
            # turn back to the originating conversation row. Format is opaque
            # (the column is plain TEXT); pick session-prefixed so a single
            # column can disambiguate across sessions on the same conv id.
            source_turn_id = (
                f"{session_id}:{conv_row_id}" if conv_row_id is not None else None
            )
            # fire-and-forget ingest. Never blocks; never raises.
            # tenka_resp passes the resolved intent so non-conversational
            # replies (web_search, set_reminder, code_executor, store_memory,
            # planner, ...) get skipped — they produce ephemeral output that
            # pollutes the graph.
            try:
                asyncio.create_task(knowledge_graph.ingest_turn(
                    transcription, "user_msg",
                    source_turn_id=source_turn_id,
                ))
                asyncio.create_task(knowledge_graph.ingest_turn(
                    clean_response, "tenka_resp",
                    reply_intent=intent_result.intent,
                    source_turn_id=source_turn_id,
                ))
            except Exception as e:
                logger.debug(f"[KG] ingest_turn dispatch failed (non-critical): {e}")

        if not _use_streaming:
            _save_turn(response_text)

        # Track conversation count for personality reflection trigger
        try:
            personality.increment_conversation_count()
        except Exception as e:
            logger.debug(f"[PERSONALITY] Counter increment failed (non-critical): {e}")

        try:
            from .personalities import get_active_loader
            if get_active_loader().get_feature_flags().get("wellbeing_checkin"):
                _wb_count = personality.get_metadata("wellbeing_checkin_counter")
                _wb_counter = int(_wb_count) if _wb_count else 0
                _wb_counter += 1
                if _wb_counter >= _WELLBEING_INTERVAL:
                    import random
                    _wb_checkin = random.choice(_WELLBEING_CHECKINS)
                    if not _use_streaming:
                        response_text = response_text.rstrip() + " " + _wb_checkin
                    personality.set_metadata("wellbeing_checkin_counter", "0")
                    logger.info(f"[WELLBEING] Check-in appended (counter reset)")
                else:
                    personality.set_metadata("wellbeing_checkin_counter", str(_wb_counter))
        except Exception as e:
            logger.debug(f"[WELLBEING] Counter failed (non-critical): {e}")

        # Step 5b: Extract and store any personal facts from this turn
        # Only on conversational intents — no point extracting from tool commands
        # Deferred for streaming path — runs after speak_streaming returns
        _facts_extracted = False
        if not _use_streaming and intent_result.intent in ("small_talk", "unknown"):
            try:
                facts = await llm.extract_facts(transcription)
                for fact in facts:
                    memory_type = await llm.ask_for_memory_type(fact["key"], fact["value"])
                    memory.save_typed_fact(
                        key=fact["key"],
                        value=fact["value"],
                        source="conversation",
                        memory_type=memory_type,
                    )
                    logger.info(f"[MEMORY] Extracted fact: {fact['key']}={fact['value']} (type={memory_type})")
                _facts_extracted = len(facts) > 0
            except Exception as e:
                logger.debug(f"Fact extraction failed (non-critical): {e}")

        # Event-driven personality trait bumps
        try:
            personality.process_turn(
                transcription, intent_result.intent, _facts_extracted
            )
        except Exception as e:
            logger.debug(f"[PERSONALITY] Event bump failed (non-critical): {e}")

        # Instant preference learning from explicit corrections
        try:
            preferences.check_for_corrections(
                transcription, intent_result.intent, intent_result.params
            )
        except Exception as e:
            logger.debug(f"Correction check failed (non-critical): {e}")

        # Step 6: TTS — speak the response with expression
        # (Skip for computer_task — the agent already speaks results via tts_func)
        if intent_result.intent != "computer_task":
            if _use_streaming:
                # Streaming path: peek at first tokens to extract emotion tag,
                # then pipe cleaned text through the streaming pipeline.
                from .io.audio.streaming import speak_streaming

                _peeked: list[str] = []
                _tag_text = ""
                emotion = "neutral"

                async for chunk in _stream:
                    _peeked.append(chunk)
                    _tag_text += chunk
                    if "]" in _tag_text:
                        parsed, clean = llm.parse_emotion_tag(_tag_text)
                        if parsed:
                            emotion = parsed
                            logger.info(f"[EMOTION] Parsed from stream: {emotion}")
                        _peeked = [clean] if clean else []
                        break
                    if len(_tag_text) > 30:
                        break

                async def _resumed_stream():
                    for chunk in _peeked:
                        if chunk:
                            yield chunk
                    async for chunk in _stream:
                        yield chunk

                unity_expression = config.UNITY_EXPRESSION_MAP.get(emotion, "neutral")
                await bridge.send_command("set_expression", value=unity_expression)

                _t0_tts = _time.monotonic()
                success, response_text = await speak_streaming(
                    _resumed_stream(), bridge, emotion=emotion
                )
                _tracker.latency_tts_ms = int((_time.monotonic() - _t0_tts) * 1000)

                if not success and not response_text:
                    await tts.speak("Sorry, something went wrong.", bridge)

                logger.info(f'Response: "{response_text}"')
                _save_turn(response_text)

                # Deferred fact extraction — runs after audio finishes
                try:
                    facts = await llm.extract_facts(transcription)
                    for fact in facts:
                        memory_type = await llm.ask_for_memory_type(fact["key"], fact["value"])
                        memory.save_typed_fact(
                            key=fact["key"], value=fact["value"],
                            source="conversation", memory_type=memory_type,
                        )
                        logger.info(f"[MEMORY] Extracted fact: {fact['key']}={fact['value']} (type={memory_type})")
                except Exception as e:
                    logger.debug(f"Fact extraction failed (non-critical): {e}")

                await bridge.send_command("set_expression", value="neutral")
                await _finish_turn(bridge, source)
            else:
                # Non-streaming path (tool results, pending states, etc.)
                parsed_emotion, response_text = llm.parse_emotion_tag(response_text)

                if parsed_emotion is not None:
                    emotion = parsed_emotion
                    logger.info(f"[EMOTION] Parsed from response: {emotion}")
                elif intent_result.intent in ("small_talk", "unknown"):
                    emotion = "neutral"
                elif "error" in response_text.lower() or "sorry" in response_text.lower() or "failed" in response_text.lower():
                    emotion = "worried"
                else:
                    emotion = "happy"

                unity_expression = config.UNITY_EXPRESSION_MAP.get(emotion, "neutral")

                await bridge.send_command("set_expression", value=unity_expression)
                _t0_tts = _time.monotonic()
                await tts.speak(response_text, bridge, emotion=emotion)
                _tracker.latency_tts_ms = int((_time.monotonic() - _t0_tts) * 1000)
                await bridge.send_command("set_expression", value="neutral")
                await _finish_turn(bridge, source)

    except Exception as e:
        _tracker.action_outcome = "failure"
        _tracker.error_class = type(e).__name__
        logger.error(f"Pipeline error: {e}", exc_info=True)
        try:
            await tts.speak("Sorry, something went wrong.", bridge)
        except Exception:
            pass
    finally:
        _tracker.save()
        _telemetry.reset_current_tracker(_tracker_token)
        _actions_module.current_grants.reset(_grants_token)
        # The turn this "studio" item started is only now actually over --
        # success, failure or an early return above all reach this
        # `finally`. Clearing the busy flag here, not at the instant
        # _StudioDispatch.submit() enqueued it, is what makes the flag mean
        # "a turn is in flight" rather than "an enqueue just happened".
        # Closing half of the turn bracket (see _publish_turn_status), before
        # the flag is released so a pane is never told it may send again
        # ahead of being told the last turn ended.
        _publish_turn_status(source, StatusPhase.IDLE)
        if source == "studio" and _studio_dispatch is not None:
            _studio_dispatch.mark_done()
        # Resume wake word detection after pipeline completes
        # (resume() auto-clears ring buffer to prevent TTS audio bleed)
        if _wake_listener:
            _wake_listener.resume()


def _grants_for_item(item: tuple) -> "tuple[frozenset[Capability], int | None]":
    """Read a queued item's grant set and its STT timing out of its third slot.

    Two producers put items on `_input_queue` and they mean different things
    by that slot, which is why this is a function and not an inline unpack:

    - Local sources ("stt", "chat") enqueue `(source, text)` or
      `(source, text, stt_ms)`. They get the full set, stated explicitly --
      the person is at this machine, and that is a deliberate grant, not the
      absence of one.
    - Studio enqueues `(source, text, grants)`, the set the authenticated
      device holds after the listener ceiling narrowed it.

    A "studio" item without a third slot -- which nothing produces today --
    gets the *empty* set, not the full one. It means the grant was lost in
    transit, and a lost grant is refused, never assumed.
    """
    extra = item[2] if len(item) > 2 else None
    if item[0] == "studio":
        return (extra if isinstance(extra, frozenset) else frozenset()), None
    return frozenset(Capability), extra


async def _process_one_queued_item(item: tuple, bridge: UnityBridge) -> None:
    """Handle exactly one item already pulled off `_input_queue`.

    Split out of the consumer loop's inline body for one reason: a studio
    turn's busy flag has to clear no matter *where* inside
    process_text_from_queue a raise happens, and three things run before
    that function's own `try:` -- constructing the TurnTracker, registering
    it with `_telemetry.set_current_tracker`, and (if a wake-word listener
    is running) pausing it. A raise in any of those three skips
    process_text_from_queue's own `finally` entirely (it is never reached),
    which used to strand `_busy=True` forever: no watchdog, no timeout, the
    Studio channel answering 409 to every message until the process
    restarted. This function's own `finally` covers the whole call --
    including everything before process_text_from_queue's internal `try:` --
    so `mark_done()` fires exactly once more here even on that failure path.

    On the ordinary success path this means `mark_done()` runs twice (once
    from inside process_text_from_queue's own `finally`, once from this
    one) -- harmless, since it only ever does `self._busy = False`, an
    idempotent assignment with no side effect to double up on.

    A second benefit of the extraction: this exact seam is now callable
    from a test with one queued item, without booting the assistant or
    running the surrounding `while not _shutdown_event.is_set()` loop at
    all.
    """
    source, text = item[0], item[1]
    grants, stt_ms = _grants_for_item(item)
    # The status bracket is mirrored here for the same reason mark_done() is,
    # and it has to be *both* halves: a raise before the inner try: leaves the
    # bus at whatever the previous turn ended on, and a lone closing IDLE
    # against that resting state dedupes to nothing (see
    # _publish_turn_status). On the ordinary path the inner pair fires first
    # and these two dedupe away -- the same harmless doubling mark_done()
    # already documents.
    _publish_turn_status(source, StatusPhase.THINKING)
    try:
        await process_text_from_queue(source, text, bridge, stt_ms=stt_ms,
                                      grants=grants)
    finally:
        _publish_turn_status(source, StatusPhase.IDLE)
        if source == "studio" and _studio_dispatch is not None:
            _studio_dispatch.mark_done()


# ─── Event Handling ──────────────────────────────────────────────────────────

# Flag to track recording state
_is_processing = False


_compression_cache: dict | None = None
_RECOMPRESS_INTERVAL = 5


def _get_personality_switch_ts() -> str | None:
    """Return the timestamp of the last personality switch, or None."""
    try:
        from .storage.db import get_db
        db = get_db()
        if db is not None:
            row = db.fetchone(
                "SELECT updated_at FROM metadata WHERE key = 'active_personality'"
            )
            if row:
                return row["updated_at"]
    except Exception:
        pass
    return None


def _strip_name_prefix(text: str) -> str:
    """Strip leading assistant name announcement from a response."""
    name = config.ASSISTANT_NAME_DISPLAY
    for prefix in (f"{name}. ", f"{name}, ", f"{name}: "):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


async def _build_conversation_messages() -> tuple[list[dict], str | None]:
    """
    Build native multi-turn messages from recent conversation history.
    Returns (messages, compressed_summary).
    """
    global _compression_cache

    turns = memory.get_recent(25)
    logger.debug(f"[CC] _build_conversation_messages: {len(turns)} turns from memory")
    if not turns:
        return [], None

    since = _get_personality_switch_ts()

    summary = None
    if len(turns) > 15:
        boundary_turn = turns[-11]
        boundary_id = boundary_turn.get("id", 0)
        cached_id = _compression_cache["last_compressed_id"] if _compression_cache else -1
        new_turns_since = boundary_id - cached_id

        if _compression_cache and new_turns_since < _RECOMPRESS_INTERVAL:
            logger.debug(f"[CC] Cache hit ({new_turns_since} new turns since last compression < {_RECOMPRESS_INTERVAL})")
            summary = _compression_cache["summary"]
            verbatim_turns = turns[-10:]
        else:
            from .llm.contracts import ask_for_context_compression
            to_compress = turns[:-10]
            logger.debug(f"[CC] Compressing {len(to_compress)} old turns, keeping {len(turns[-10:])} verbatim")
            summary = await ask_for_context_compression(to_compress)
            logger.debug(f"[CC] Compression result: {summary[:120] if summary else '(empty)'}...")
            _compression_cache = {
                "summary": summary,
                "last_compressed_id": boundary_id,
            }
            verbatim_turns = turns[-10:]
    else:
        logger.debug(f"[CC] Below threshold ({len(turns)} <= 15), no compression")
        verbatim_turns = turns

    messages: list[dict] = []
    for t in verbatim_turns:
        messages.append({"role": "user", "content": t["user_input"]})
        if since and t.get("timestamp", "") < since:
            messages.append({"role": "assistant", "content": "(responded)"})
        else:
            messages.append({
                "role": "assistant",
                "content": _strip_name_prefix(t["response"]),
            })

    return messages, summary


def _build_facts_context() -> str:
    """
    Build a known facts string from stored user facts.
    Returns empty string if no facts or on any error.
    """
    try:
        # Search broadly for all user facts
        facts = memory.search_facts("user_")
        if not facts:
            return ""

        lines = ["KNOWN FACTS ABOUT THE USER:"]
        seen_keys = set()
        for f in facts:
            key = f["key"]
            if key not in seen_keys:
                seen_keys.add(key)
                lines.append(f"  {key}: {f['value']}")

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""

async def on_unity_event(event: dict, bridge: UnityBridge):
    """
    Handle an event received from Unity via the bridge.

    Events:
      {"event": "start_listening"}  — User pressed the V button in Unity
      {"event": "stop_listening"}   — User released the V button (or toggle off)
      {"event": "avatar_clicked"}   — User clicked on the avatar
    """
    global _is_processing

    event_name = event.get("event", "")

    if event_name == "start_listening":
        if _is_processing:
            logger.warning("Already processing, ignoring start_listening")
            return

        if not recorder.is_recording:
            recorder.start()
            await bridge.send_command("set_expression", value="surprised")

    elif event_name == "stop_listening":
        if recorder.is_recording:
            _is_processing = True
            try:
                await run_pipeline(bridge)
            finally:
                _is_processing = False

    elif event_name == "avatar_clicked":
        logger.info("Avatar clicked — waving!")
        await bridge.send_command("play_animation", name="wave")

    elif event_name == "chat_message":
        text = event.get("text", "")
        if text.strip():
            logger.debug(f"Adding chat message to input queue: {text}")
            _input_queue.put(("chat", text.strip()))

    else:
        logger.debug(f"Unknown event: {event_name}")


# ─── Keyboard Listener (fallback when Unity not connected) ───────────────────


async def keyboard_listener(bridge: UnityBridge):
    """
    Listen for the V key press in the terminal.
    Press V once to start recording, press V again to stop and process.
    This is for testing without Unity running.
    """
    global _is_processing

    logger.info(
        f"Keyboard listener ready — press '{config.PUSH_TO_TALK_KEY.upper()}' "
        "to start/stop recording"
    )

    try:
        from pynput import keyboard as kb

        loop = asyncio.get_running_loop()
        talk_key_raw = config.PUSH_TO_TALK_KEY.lower().strip()

        # Resolve the configured key into either a single character or a
        # pynput Key enum value. Single chars match key.char (letters,
        # digits). Names like "home", "end", "f1", "page_up" match the
        # Key.NAME enum on pynput. Unknown special-key names fall back to
        # the literal string match (won't fire — user gets a no-op listener).
        if len(talk_key_raw) == 1:
            talk_key_kind = "char"
            talk_key_value = talk_key_raw
        else:
            special = getattr(kb.Key, talk_key_raw, None)
            if special is not None:
                talk_key_kind = "key"
                talk_key_value = special
            else:
                talk_key_kind = "char"
                talk_key_value = talk_key_raw  # fallback; won't match
                logger.warning(
                    f"PUSH_TO_TALK_KEY='{talk_key_raw}' is not a single char "
                    f"and not a recognized pynput Key (home, end, f1-12, page_up, "
                    f"page_down, insert, delete, etc.). Push-to-talk disabled."
                )

        def _matches(key) -> bool:
            if talk_key_kind == "char":
                return hasattr(key, "char") and key.char is not None and key.char.lower() == talk_key_value
            return key == talk_key_value

        def on_press(key):
            try:
                from .automation import vision as computer_agent, native as app_automation
                # Suppress while either subsystem is synthesizing keystrokes
                # — typing "Vestibulum" shouldn't toggle push-to-talk.
                if computer_agent._agent_typing or app_automation._app_typing:
                    return

                if _matches(key):
                    # Barge-in: stop streaming THEN start recording (sequenced
                    # on the event loop to avoid concurrent sounddevice access)
                    from .io.audio.streaming import is_speaking, stop_streaming
                    if is_speaking():
                        async def _barge_in():
                            logger.info("[BARGE-IN] Stopping streaming playback")
                            try:
                                await stop_streaming()
                            except Exception as e:
                                logger.debug(f"[BARGE-IN] stop_streaming error: {e}")
                            if not recorder.is_recording:
                                recorder.start()
                                logger.info(f"[MIC] Press {config.PUSH_TO_TALK_KEY.upper()} again to stop recording")
                        loop.call_soon_threadsafe(
                            lambda: asyncio.ensure_future(_barge_in())
                        )
                        return

                    if _is_processing:
                        return

                    if not recorder.is_recording:
                        recorder.start()
                        logger.info(f"[MIC] Press {config.PUSH_TO_TALK_KEY.upper()} again to stop recording")
                    else:
                        # Stop recording and run pipeline in the async loop
                        loop.call_soon_threadsafe(
                            lambda: asyncio.ensure_future(_process_keyboard(bridge))
                        )
            except Exception:
                pass

        listener = kb.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

        # Keep running until cancelled
        while True:
            await asyncio.sleep(1)

    except ImportError:
        logger.warning(
            "pynput not installed — keyboard shortcut disabled. "
            "Install with: pip install pynput"
        )
        # Fall back to simple input() loop
        while True:
            await asyncio.sleep(1)


async def _process_keyboard(bridge: UnityBridge):
    """Process a keyboard-triggered recording."""
    global _is_processing
    _is_processing = True
    try:
        await run_pipeline(bridge)
    finally:
        _is_processing = False


# ─── Follow-up Listen ────────────────────────────────────────────────────────

# Whisper commonly hallucinates these phrases on silence / background noise.
# NOTE: "yes", "no", "ok", "okay" are intentionally NOT here — they're valid
# short responses during dialogue. Filtering is done via speech_secs instead.
_WHISPER_HALLUCINATIONS = frozenset({
    "thank you", "thanks", "thank you.", "thanks.", "thanks!",
    "thank you!", "thank you for watching", "thank you for watching.",
    "you", "bye", "bye.", "goodbye", "goodbye.", "um", "uh", ".",
    "subtitles by", "[silence]", "(silence)", "(music)",
})


async def _follow_up_listen() -> tuple[str, int | None]:
    """
    After TTS finishes, listen until the user stops talking (silence-based),
    with a hard cap at FOLLOW_UP_LISTEN_SECONDS.

    Returns a (transcription, stt_ms) tuple. transcription is "" on silence,
    in which case stt_ms is None. Wake word listener must be paused.
    """
    if recorder.is_recording:
        return "", None
    await asyncio.sleep(0.3)  # tail clearance — avoid capturing TTS echo
    loop  = asyncio.get_running_loop()
    audio, speech_secs = await loop.run_in_executor(
        None, record_until_silence, config.FOLLOW_UP_LISTEN_SECONDS
    )
    if len(audio) == 0 or speech_secs < 0.05:
        return "", None
    _stt_start = _time.monotonic()
    text = transcribe(audio).strip()
    _stt_ms = int((_time.monotonic() - _stt_start) * 1000)
    if not text:
        return "", None
    if text.lower().rstrip(" .,!?") in _WHISPER_HALLUCINATIONS:
        logger.debug(f"[FOLLOWUP] Hallucination filtered: '{text}'")
        return "", None
    logger.info(f"[FOLLOWUP] Heard: '{text}' (stt={_stt_ms}ms)")
    return text, _stt_ms


async def _finish_turn(bridge: UnityBridge, source: str) -> None:
    """
    Called after every assistant response. Opens a follow-up listen window so the
    user can reply without saying the wake word again. Then resumes the wake
    word listener (idempotent — safe even if called before the finally block).

    `source` is required and has no default. It is not decoration: opening the
    follow-up window records from the local microphone and re-queues whatever
    it hears as `("stt", ...)`, and `_grants_for_item` hands an `stt` item
    `frozenset(Capability)` -- the full local grant set. So this function is
    the one place in the tree where a turn can *mint privilege it did not
    arrive with*, and a caller that does not say which turn it is finishing
    would be minting it blind. A default here would be the same mistake
    `ChatDispatch.submit()` refuses to make with its grants parameter.

    The attack it closes: a remote Studio message that merely matched the
    teach-trigger regex reached a `_finish_turn(bridge)` call, which opened the
    microphone, waited for the operator to say anything at all in the room, and
    ran that utterance with every capability there is. The attacker speaks
    nothing and holds only CHAT_SEND. Remote hot mic and remote-to-local
    privilege escalation in one move.
    """
    if source not in _LOCAL_SOURCES:
        # Not an error and not worth a refusal message -- a remote turn simply
        # has no follow-up window, because there is no one at the microphone
        # who asked for it. The wake listener still resumes: the turn's own
        # `finally` does it too, and doing it here keeps this function's
        # postcondition ("the listener is running again") true on both paths.
        logger.debug(f"[TURN] No follow-up listen for non-local source {source!r}")
        if _wake_listener:
            _wake_listener.resume()
        return
    followup, stt_ms = await _follow_up_listen()
    if _wake_listener:
        _wake_listener.resume()
    if followup:
        _input_queue.put(("stt", followup, stt_ms))


# ─── Wake Word Handler ────────────────────────────────────────────────────────


async def _on_wake_word_detected(bridge: UnityBridge):
    """
    Called when the wake word is detected.
    Verifies speaker from wake word audio before recording.
    Starts recording for a set duration, then runs the pipeline.
    """
    global _is_processing

    if _is_processing:
        logger.debug("Already processing, ignoring wake word")
        return

    # IMMEDIATELY pause wake word detection to prevent re-triggers
    # during TTS, verification, and recording. Every exit path must resume.
    if _wake_listener:
        _wake_listener.pause()

    # Wake word verification DISABLED.
    # A short wake utterance (e.g. "Hey TENKA", ~0.8s) is too short for reliable
    # ECAPA embeddings. Owner scores: 0.19-0.43 (inconsistent). Friend scores: -0.06-0.13.
    # Overlap zone makes it unreliable. Speaker verification runs on the
    # COMMAND audio instead (5s, natural speech → scores 0.55-0.65 for
    # owner, -0.06-0.13 for stranger). That gate is reliable.
    #
    # The ring buffer and get_recent_audio() are kept for future use
    # if a better short-utterance model becomes available.

    # Suppress recording worker from saving wake word audio as a chunk
    from . import recording as recording_module
    recording_module.suppress_for(8.0)

    logger.info("[MIC] Wake word activated — recording...")

    # Notify Unity
    await bridge.send_command("set_expression", value="surprised")

    # Give audio feedback so the user knows we're listening
    await tts.speak("yes!", bridge, emotion="excited")

    # Start recording
    recorder.start()

    # Record for the configured duration (auto-stop)
    await asyncio.sleep(config.WAKE_WORD_RECORD_SECONDS)

    _is_processing = True
    try:
        await run_pipeline(bridge, from_wake_word=True)
    finally:
        _is_processing = False


# ─── Main ────────────────────────────────────────────────────────────────────


async def async_main():
    """Main async entry point."""
    global _wake_listener, _session_resume_context

    logger.info("=" * 60)
    logger.info("  TENKA — Voice Assistant (Python)")
    logger.info("=" * 60)
    logger.info(f"STT backend:   {config.STT_BACKEND}")
    logger.info(f"TTS voice:     {config.TTS_VOICE}")
    logger.info(f"Wake word:     {'enabled' if config.WAKE_WORD_ENABLED else 'disabled'}")
    logger.info(f"LLM (Groq):    {'configured (' + str(len(llm.GROQ_KEYS)) + ' key(s))' if llm.GROQ_KEYS else 'NOT SET'}")
    logger.info(f"LLM (Cerebras):{'configured' if config.CEREBRAS_API_KEY else 'NOT SET'}")
    logger.info(f"LLM (Ollama):  {config.OLLAMA_URL} / {config.OLLAMA_MODEL}")
    logger.info(f"Sandbox dir:   {config.SANDBOX_DIR}")
    logger.info(f"Recording:     VAD-chunked, silence gap={recording.VAD_SILENCE_GAP}s, summary threshold={recording.SUMMARY_THRESHOLD} chunks")
    logger.info("")

    # Ensure sandbox directories exist
    config.SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)

    # Warm the Playwright driver FIRST — before any other init. Background:
    # spawning the bundled node.exe driver later in startup hangs the IPC
    # handshake (root cause TBD — possibly cgo from neonize, audio device
    # threads, or pytorch's thread pools interfering with subprocess pipes).
    # Doing it first proves cleanliness; if THIS still hangs, the conflict is
    # at module-import time, which the diagnostic logs in warmup_driver will
    # surface.
    try:
        from .automation.browser import automation as browser_automation
        await browser_automation.warmup_driver()
    except Exception as _e:
        logger.debug(f"[main] Playwright warmup skipped: {_e}")

    # Non-blocking CDP probe. We DO NOT gate startup on CDP being
    # live — TENKA must boot when Chrome is closed too. The probe just warms
    # the cache so the first browser task knows whether to attach or use
    # bundled. Cost: ~5ms when port is closed, ~30ms when Chrome answers.
    # Schedule as a fire-and-forget task; we don't await it here.
    try:
        from .automation.browser import cdp as browser_cdp
        asyncio.create_task(browser_cdp.cdp_health_probe(timeout=0.5))
    except Exception as _e:
        logger.debug(f"[main] CDP probe schedule skipped: {_e}")

    # Auto-start whisper.cpp server if needed
    _start_whisper_cpp_server()

    # Initialize TTS eagerly (avoids first-use delay)
    tts.init_tts()

    # Calibrate silence-detection threshold from ambient noise
    calibrate_noise_floor()

    # Initialize speaker verification
    if config.SPEAKER_VERIFY_ENABLED:
        speaker_verify.init_speaker_model()
        logger.info(
            f"Speaker verify: "
            f"{'enrolled' if speaker_verify.is_enrolled() else 'not enrolled'}"
        )

    # Shared Database for all stores. config.py may have already
    # initialized this during its early import — init_db is idempotent.
    from .storage.db import init_db
    init_db(config.SANDBOX_DIR / "memory" / "tenka.db")

    # Initialize conversation memory (requires init_db above)
    memory.init_memory()
    knowledge_graph.init_kg()
    # Clean up expired memory facts on startup
    try:
        expired = memory.cleanup_expired()
        if expired:
            logger.info(f"[MEMORY] Startup cleanup: removed {expired} expired fact(s)")
    except Exception as e:
        logger.debug(f"[MEMORY] Startup cleanup failed (non-critical): {e}")
    try:
        repo = knowledge_graph._get_repo()
        if repo is not None:
            kg_count = repo.cleanup_expired_facts()
            if kg_count:
                logger.info(f"[KG] expired-fact cleanup removed {kg_count}")
    except Exception as e:
        logger.debug(f"[KG] cleanup_expired_facts failed: {e}")

    # Prune expired automation cache entries
    try:
        from assistant.automation.step_cache import cleanup_expired
        removed = cleanup_expired(max_age_days=30)
        if removed:
            logger.info(f"[STARTUP] Pruned {removed} expired automation cache entries")
    except Exception as e:
        logger.debug(f"[STARTUP] Automation cache cleanup skipped: {e}")

    # manifest store + registry singleton
    # Pre-declared so the daily-reset loop closure always finds _vision_cap bound
    # even if the try block below bails before reaching construction.
    _vision_cap = None
    try:
        from .automation.manifest_store import ManifestStore
        from .automation.manifest_registry import init_singleton as init_manifest_registry
        from .storage.repos.app_manifest_index import AppManifestIndexRepo
        from .storage.db import get_db

        _db = get_db()
        if _db is not None:
            config.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
            _manifest_index_repo = AppManifestIndexRepo(_db._conn)
            _manifest_store = ManifestStore(
                manifests_dir=config.MANIFESTS_DIR,
                index_repo=_manifest_index_repo,
            )
            _manifest_store.scan_and_index()
            init_manifest_registry(
                store=_manifest_store, index_repo=_manifest_index_repo,
            )
            logger.info(
                f"[manifest] Manifest dir: {config.MANIFESTS_DIR}; "
                f"loaded {len(_manifest_store.all())} manifest(s)"
            )

            # dispatcher — wired to the registry + a live native handle.
            # The terminator_provider lambda runs at dispatch time (not now), so
            # late population of native._desktop is fine as long as the singleton
            # is up before the first dispatch call.
            from .automation.manifest_dispatcher import ManifestDispatcher
            from .automation.healer import Healer
            from .automation.vision_cap import VisionCapTracker
            from .automation import manifest_registry as _mreg
            from .automation import manifest_runtime
            from .automation import native as _native_automation

            _live_terminator_adapter: manifest_runtime._TerminatorAdapter | None = None

            def _get_live_terminator():
                """Return a TerminatorLike-compatible handle wrapping native.py.

                Cached for the lifetime of the process. Adapter is a stub for
                v1 — methods raise NotImplementedError until wired during the
                manifest-based live-test follow-up. Tests inject FakeTerminator and
                never exercise this path.
                """
                nonlocal _live_terminator_adapter
                if _live_terminator_adapter is None:
                    desktop = _native_automation.ensure_desktop()
                    _live_terminator_adapter = manifest_runtime._TerminatorAdapter(desktop)
                return _live_terminator_adapter

            _vision_cap = VisionCapTracker(_db._conn)
            _manifest_healer = Healer(
                store=_manifest_store,
                terminator_provider=_get_live_terminator,
                vision_cap=_vision_cap,
            )

            _manifest_registry = _mreg.get_singleton()
            if _manifest_registry is not None:
                _manifest_dispatcher = ManifestDispatcher(
                    registry=_manifest_registry,
                    store=_manifest_store,
                    terminator_provider=_get_live_terminator,
                    healer=_manifest_healer,
                )
                manifest_runtime.init_dispatcher(_manifest_dispatcher)
                logger.info("[manifest] dispatcher initialized with healer")
                logger.info(
                    "[manifest] _TerminatorAdapter fully live "
                    "(send_key/find_element/click + enumerate_descendants/screenshot/element_at_point); "
                    "healer tier-1 + tier-2 enabled"
                )
    except Exception as e:
        logger.warning(f"[manifest] Manifest store init failed (non-critical): {e}")

    # Initialize personality state
    personality.init_personality_db()

    preferences.init_preference_db()
    shortcuts.init_shortcut_db()
    procedures.init_procedure_db()

    # Runtime settings store. Init BEFORE reload so persisted values
    # override the defaults captured when config.py was first imported.
    settings.init_settings_db()
    config.reload_runtime_settings()

    # Initialize session continuity
    from . import session as session_mod
    session_mod.init_session_db()
    _session_id = session_mod.start_session()
    await session_mod.recover_crashed_session()
    _session_resume_context = session_mod.get_resume_context()
    if _session_resume_context:
        logger.info("[SESSION] Resume context loaded")

    from .core.asyncio_utils import set_main_loop
    set_main_loop(asyncio.get_running_loop())

    # ─── Cursor visibility overlay + universal abort ───────────────────────────
    _overlay_loop = asyncio.get_running_loop()
    abort.on_abort(lambda _reason: asyncio.run_coroutine_threadsafe(
        stop_streaming(), _overlay_loop
    ))
    # Surface a red "Stopped" pill the moment ESC fires so the user gets
    # immediate visual confirmation (overlay auto-hides after ~1.2s).
    abort.on_abort(lambda _reason: status.set(StatusPhase.STOPPED))
    esc_monitor.start()
    overlay_manager.start()
    status.set(StatusPhase.IDLE)

    proactive.start_analyzer()

    # Initialize reminders
    reminders.start()

    # scheduled conditional tasks
    from . import scheduler
    scheduler.start(loop=asyncio.get_running_loop())

    # event-driven monitors
    from .automation.event_bus import event_bus as _event_bus
    try:
        _event_bus.start(loop=asyncio.get_running_loop())
    except Exception as e:
        logger.warning("[event-monitor] EventBus failed to start (non-critical): %s", e)

    # Telemetry
    _telemetry.init_telemetry_db()
    try:
        _deleted = _telemetry._get_repo().cleanup(retention_days=config.TELEMETRY_RETENTION_DAYS)
        if _deleted:
            logger.info(f"[TELEMETRY] Startup cleanup: removed {_deleted} event(s)")
    except Exception as e:
        logger.debug(f"[TELEMETRY] Startup cleanup failed (non-critical): {e}")

    # Background task: clean expired memory facts every hour
    async def _memory_cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            try:
                count = memory.cleanup_expired()
                if count:
                    logger.info(f"[MEMORY] Hourly cleanup: removed {count} expired fact(s)")
            except Exception as e:
                logger.debug(f"[MEMORY] Hourly cleanup failed (non-critical): {e}")
            try:
                repo = knowledge_graph._get_repo()
                if repo is not None:
                    kg_count = repo.cleanup_expired_facts()
                    if kg_count:
                        logger.info(f"[KG] expired-fact cleanup removed {kg_count}")
            except Exception as e:
                logger.debug(f"[KG] cleanup_expired_facts failed: {e}")

    asyncio.get_running_loop().create_task(_memory_cleanup_loop())

    # Background task: the TENKA Studio daemon, off by default. A second
    # asyncio task on this same loop rather than a second process, because
    # that is what keeps it on the one SQLite connection the rest of the
    # assistant shares. Building/starting it lives in _start_studio_daemon()
    # so a test can drive that exact sequence without booting the assistant.
    _studio_task: asyncio.Task | None = None
    if config.STUDIO_API_ENABLED:
        _studio_task = await _start_studio_daemon()

    # Background task: reset manifest-based daily vision cap when the local date rolls over.
    async def _vision_cap_daily_reset_loop():
        """Hourly check; reset the manifest-based vision cap when the local date rolls over."""
        from datetime import datetime
        last_day = datetime.now().date()
        while True:
            await asyncio.sleep(3600)
            try:
                today = datetime.now().date()
                if today != last_day:
                    if _vision_cap is not None:
                        _vision_cap.reset_for_new_day()
                        logger.info("[manifest] Vision cap reset for new day")
                    last_day = today
            except Exception as e:
                logger.debug(f"[manifest] Daily vision-cap reset failed (non-critical): {e}")

    if _vision_cap is not None:
        asyncio.get_running_loop().create_task(_vision_cap_daily_reset_loop())

    # Start messaging bridge (WhatsApp, Telegram, etc.)
    messaging_bridge.start()

    # Auto-connect messaging services with saved sessions
    messaging_bridge.auto_connect_services()

    # pre-warm embedding model at startup
    memory.warm_embed_model()

    # Re-apply third-party log silencing after model loads.
    # speechbrain's get_logger() and sentence_transformers both call setLevel(INFO)
    # during model loading, overriding the module-level settings above.
    _silence_third_party_loggers()

    # Create the bridge — terminal-only mode swaps in a no-op stub.
    bridge = UnityBridge() if config.UNITY_ENABLED else NullBridge()

    # Set up the event callback — wraps on_unity_event to pass bridge
    async def event_handler(event):
        await on_unity_event(event, bridge)

    # Start the bridge servers
    await bridge.start(event_callback=event_handler)

    # Start the wake word listener (if enabled)
    if config.WAKE_WORD_ENABLED:
        loop = asyncio.get_running_loop()

        def wake_word_callback():
            """Called from the wake word background thread."""
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(_on_wake_word_detected(bridge))
            )

        _wake_listener = WakeWordListener(on_wake_word=wake_word_callback)
        _wake_listener.start()

    # Warm the geolocation cache (non-blocking, 3s timeout)
    try:
        from .core.geolocation import detect_region
        await detect_region()
    except Exception:
        pass

    logger.info("")
    logger.info("Ready! Waiting for Unity to connect, say the wake word, or press V to talk...")
    logger.info("Or type a message in the console and hit Enter.")
    logger.info("")

    # Start the chat thread
    chat_thread = threading.Thread(target=_chat_input_loop, args=(_input_queue,), daemon=True)
    chat_thread.start()

    # Run the keyboard listener concurrently
    try:
        loop = asyncio.get_running_loop()
        keyboard_task = loop.create_task(keyboard_listener(bridge))
        
        # Main processing loop
        while not _shutdown_event.is_set():
            try:
                item = _input_queue.get_nowait()
                await _process_one_queued_item(item, bridge)
            except queue.Empty:
                # Check if the recording worker detected a voice stop command
                if recording.voice_stop_requested():
                    logger.info("[RECORDING] Voice stop detected — injecting stop command")
                    _input_queue.put(("stt", "stop recording"))
                    continue

                # Drain background file search results
                try:
                    from .actions import _search_result_queue
                    search_result = _search_result_queue.get_nowait()
                    # Don't interrupt active recording
                    if not recording.is_active():
                        logger.info(f"[FILE] Delivering background search result")
                        from .personalities import get_active_loader as _get_loader
                        if _get_loader().get_emotion_mode() == "neutral":
                            emotion = "neutral"
                        else:
                            emotion = await llm.classify_emotion(search_result)
                        await tts.speak(search_result, bridge, emotion=emotion)
                        memory.save_turn(
                            "[background search]", "file_task",
                            search_result,
                            session_mod.get_current_session_id(),
                        )
                    else:
                        # Re-queue — deliver after recording stops
                        _search_result_queue.put(search_result)
                except queue.Empty:
                    pass

                # Drain proactive nudges — defer if user is active or TTS is playing
                try:
                    item = proactive.get_queue().get_nowait()
                    if isinstance(item, tuple):
                        nudge, emotion = item
                    else:
                        nudge, emotion = item, None
                    from .io.audio.streaming import is_speaking as streaming_speaking
                    tts_busy = tts.is_speaking() or streaming_speaking()
                    if (config.PROACTIVE_MODE == "always"
                            or (not _is_processing and not tts_busy
                                and not recorder.is_recording)):
                        logger.info(f"[MONITOR] Delivering: {nudge[:80]}")
                        if emotion is None:
                            from .personalities import get_active_loader as _get_loader2
                            if _get_loader2().get_emotion_mode() == "neutral":
                                emotion = "neutral"
                            else:
                                emotion = await llm.classify_emotion(nudge)
                        await tts.speak(nudge, bridge, emotion=emotion)
                        memory.save_turn(
                            "[proactive]", "proactive", nudge,
                            session_mod.get_current_session_id(),
                        )
                    else:
                        proactive.get_queue().put(item)
                except queue.Empty:
                    pass

                # Drain and announce incoming message notifications
                try:
                    await _drain_and_announce_notifications(bridge)
                except Exception as e:
                    logger.debug(f"[NOTIFY] Error in notification drain: {e}")

                await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        pass
    finally:
        # Save session snapshot before shutdown
        try:
            sid = session_mod.get_current_session_id()
            if sid:
                rows = memory._get_repo()._db.fetchall(
                    "SELECT user_input, response, session_id FROM conversations "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (sid,),
                )
                if rows:
                    turns = [{"user_input": r["user_input"], "response": r["response"]} for r in rows]
                    await session_mod.save_snapshot(turns)
        except Exception as e:
            logger.debug(f"[SESSION] Shutdown snapshot failed (non-critical): {e}")
        session_mod.end_session()

        if 'keyboard_task' in locals():
            keyboard_task.cancel()
        # Clean up
        if _wake_listener:
            _wake_listener.stop()
        proactive.stop_analyzer()
        reminders.stop()
        scheduler.stop()
        try:
            _event_bus.stop()
        except Exception:
            pass
        messaging_bridge.stop()
        _stop_whisper_cpp_server()

        # ─── manifest-based final promotion cycle ──────────────────────────────────
        # Run one last promotion pass on shutdown so freshly-learned
        # behaviours land in YAML before the process exits. All failures
        # are swallowed — shutdown must never get stuck on promotion.
        try:
            from .automation import promoter as _promoter_mod
            from .automation.manifest_registry import (
                get_singleton as _get_manifest_registry,
            )
            from .automation.promoter import Promoter
            from .storage.db import get_db as _get_db_for_shutdown
            from .storage.repos.automation_cache import AutomationCacheRepo

            _registry = _get_manifest_registry()
            _shutdown_db = _get_db_for_shutdown()
            if _registry is not None and _shutdown_db is not None:
                if _promoter_mod.is_promotion_in_flight():
                    logger.info(
                        "[manifest] Shutdown cycle skipped — promotion already in flight."
                    )
                else:
                    _promoter_mod._set_in_flight(True)
                    try:
                        _promoter = Promoter(
                            automation_cache_repo=AutomationCacheRepo(_shutdown_db),
                            manifest_store=_registry.store,
                        )
                        _summary = await _promoter.run_once()
                        logger.info(f"[manifest promote] shutdown summary: {_summary}")
                    except Exception as e:
                        logger.warning(
                            f"[manifest promote] shutdown cycle failed: {e}",
                            exc_info=True,
                        )
                    finally:
                        _promoter_mod._set_in_flight(False)
        except Exception as e:
            logger.warning(f"[manifest promote] shutdown cycle failed: {e}")

        # ─── Cursor visibility overlay shutdown ────────────────────────────────────
        try:
            status.set(StatusPhase.IDLE)
        except Exception:
            pass
        overlay_manager.stop()
        esc_monitor.stop()

        # ─── Studio daemon shutdown ─────────────────────────────────────────────
        # This block runs on every *orderly* exit from the loop above --
        # Ctrl+C, the "shutdown" voice intent, even an exception caught
        # somewhere upstream that unwinds through this `finally` -- which
        # used to mean `shutdown_studio_api()` (server.shutdown, the kill
        # switch that rotates the instance secret via vault.reset()) fired
        # unconditionally too. That is backwards from what a kill switch is
        # for: a phone paired yesterday would have to be re-paired after
        # every ordinary restart, and the fresh token that is meant to print
        # once, at first pairing, would print on every boot instead. A real
        # crash (killed from Task Manager, a power loss) never reaches this
        # `finally` at all, so the previous code had the two situations
        # exactly inverted -- a clean exit destroyed pairing state that a
        # crash would have preserved.
        #
        # Only stopping serving belongs here. `_stop_studio_daemon` cancels
        # the task and awaits it, which is what actually releases the
        # listening port (see server.serve()'s own comment on why a bare
        # cancel alone is not enough) -- it never touches the vault, so every
        # device issued before this restart still verifies after it.
        # server.shutdown() (the kill switch) stays fully defined and
        # covered by its own tests in test_api_hardening.py and
        # test_api_server_lifecycle.py; nothing in this milestone gives it a
        # deliberate trigger yet (a future "revoke every Studio device" admin
        # action would call it explicitly), and it must not be reachable from
        # a normal exit path in the meantime.
        await _stop_studio_daemon(_studio_task)

        await bridge.stop()
        logger.info("Voice Assistant shut down")


def main():
    """Synchronous entry point."""
    # Force ProactorEventLoopPolicy on Windows — Playwright requires it for
    # subprocess IPC (driver spawn). If a third-party library somehow set
    # SelectorEventLoopPolicy during import, async_playwright().start() hangs
    # indefinitely on the IPC handshake. This is a no-op on non-Windows.
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logger.info(f"[main] asyncio policy: {type(asyncio.get_event_loop_policy()).__name__}")
        except Exception as _e:
            logger.warning(f"[main] Could not set ProactorEventLoopPolicy: {_e}")

    # Handle Ctrl+C gracefully — first press triggers clean shutdown,
    # second press forces immediate exit (in case cleanup hangs).
    def signal_handler(sig, frame):
        if _shutdown_event.is_set():
            logger.info("Force shutdown")
            _stop_whisper_cpp_server()
            sys.exit(1)
        logger.info("Shutting down gracefully... (Ctrl+C again to force)")
        _stop_whisper_cpp_server()
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_shutdown_event.set)
        except RuntimeError:
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        _stop_whisper_cpp_server()
        logger.info("Goodbye!")

    # Force-exit: the daemon chat-input thread blocks on input() and
    # keeps the process alive; subprocess __del__ triggers "Event loop
    # is closed" errors during GC. All cleanup ran in async_main's
    # finally block, so it's safe to terminate immediately.
    os._exit(0)


if __name__ == "__main__":
    main()