"""
core/intent_capabilities.py — Which capability each intent requires.

Pure data module, same shape as `core/intent_scopes.py`: one import, no logic.
`actions/__init__.py` reads it at the single dispatch choke point, so it must
stay cheap and dependency-free.

**The default is the strongest capability, not the weakest.** An intent with no
row here requires `EXECUTE`. A new intent added to `config.INTENTS` without
being classified keeps working on the local path (the operator's own grant set
holds everything) and is refused over every transport until someone decides
what it is. Failing closed silently is still failing closed, but it is silent,
so `tests/test_6a5_stream_a.py` turns a missing row into a loud test failure as
well.

The classification, in one sentence each:

- `EXECUTE` — runs code or drives the machine, *or* installs something that
  will. The four `manage_*` intents are in this group for the second reason: a
  monitor's `_fire_action` calls `execute("code_executor", ...)` directly, so
  gating the installed thing and not the installer would be theatre.
- `FILES` — reads or writes the user's files.
- `SCREEN` — captures the screen or the camera, including the recording
  intents, whose transcripts are made of exactly that.
- `SYSTEM_CONTROL` — changes how the machine itself is configured. Voice
  enrollment is here because it rewrites who she will obey.
- `CHAT_SEND` — conversation and the reads that go with it. Everything whose
  worst case is that she says something.

`browser_action` and `app_action` are registered handlers but internal routing
targets rather than user-facing intents; they are correctly absent from
`config.INTENTS` and therefore absent here. They are reached only from an
intent that already passed the gate.
"""

from .capabilities import Capability

REQUIRED_CAPABILITY: dict[str, Capability] = {
    # ── Runs code, or installs something that will ───────────────────────
    "code_executor": Capability.EXECUTE,
    "computer_task": Capability.EXECUTE,
    "find_and_click": Capability.EXECUTE,
    "manifest_dispatch": Capability.EXECUTE,
    "planner": Capability.EXECUTE,
    "browser_extension_setup": Capability.EXECUTE,
    # Listing tabs reveals what the user is reading, and closing one
    # destroys a page she cannot get back. Both sit behind EXECUTE
    # rather than splitting the read from the write: one intent, one
    # capability, and the stronger of the two.
    "browser_tabs": Capability.EXECUTE,
    "manage_procedure": Capability.EXECUTE,
    "manage_monitor": Capability.EXECUTE,
    "manage_schedule": Capability.EXECUTE,
    "manage_shortcut": Capability.EXECUTE,
    "shutdown": Capability.EXECUTE,
    "manage_backup": Capability.EXECUTE,

    # ── The user's files ─────────────────────────────────────────────────
    "file_task": Capability.FILES,

    # ── The screen and the camera ────────────────────────────────────────
    "read_screen": Capability.SCREEN,
    "camera_look": Capability.SCREEN,
    "start_recording": Capability.SCREEN,
    "stop_recording": Capability.SCREEN,
    "get_recording": Capability.SCREEN,
    "summarize_recording": Capability.SCREEN,
    "meet_face": Capability.SCREEN,
    "recognize_face": Capability.SCREEN,
    "forget_face": Capability.SCREEN,

    # ── How the machine is configured ────────────────────────────────────
    "enroll_voice": Capability.SYSTEM_CONTROL,
    "forget_voice": Capability.SYSTEM_CONTROL,

    # ── Conversation ─────────────────────────────────────────────────────
    "small_talk": Capability.CHAT_SEND,
    "unknown": Capability.CHAT_SEND,
    "get_time": Capability.CHAT_SEND,
    "web_search": Capability.CHAT_SEND,
    "browse_url": Capability.CHAT_SEND,
    "open_browser": Capability.CHAT_SEND,
    "create_note": Capability.CHAT_SEND,
    "memory_query": Capability.CHAT_SEND,
    # The intent itself costs no more than being able to talk. The
    # *facts* are gated individually in `brain/selfknowledge.py`, per
    # fact class, against the route that already publishes the same
    # information -- putting the whole intent behind one capability
    # would either hide the architecture (public in this repository)
    # or expose the transport list (loopback-only, SYSTEM_CONTROL).
    "self_knowledge": Capability.CHAT_SEND,
    "store_memory": Capability.CHAT_SEND,
    "forget_memory": Capability.CHAT_SEND,
    "set_reminder": Capability.CHAT_SEND,
    "cancel_reminder": Capability.CHAT_SEND,
    "hide_avatar": Capability.CHAT_SEND,
    "show_avatar": Capability.CHAT_SEND,
}

# Unlisted intents require the strongest capability. See the module docstring.
DEFAULT_REQUIRED: Capability = Capability.EXECUTE


# ─── Does the effect outlive the turn? ───────────────────────────────────────
#
# `REQUIRED_CAPABILITY` above answers "what does this intent cost". This pair
# answers a second question the raise mechanism made necessary: "does spending
# it leave something behind that runs later?"
#
# A raise is deliberately time-bounded -- minted at the keyboard, scoped to one
# device and one transport, expiring. But `manage_monitor` and friends install
# something that `automation/event_bus.py` and `scheduler.py` later run with
# `LOCAL_GRANTS`, on the stated argument that whoever installed it already held
# `EXECUTE`. That is true for as long as the raise lasts and false afterwards,
# so a half-hour raise could be converted into permanent local execution by
# installing a monitor with it. `actions.durable_capability_refusal` is the
# check; these two sets are the data it reads.
#
# **Exhaustive, with no default in either direction.** This is the one place a
# strong default does not work, and the reason is worth stating because the
# obvious choice is wrong both ways:
#
#   * defaulting to "persists" would refuse `code_executor` to a raised
#     device, which destroys the entire purpose of a raise -- running code on
#     a vetted machine is the thing it exists to permit;
#   * defaulting to "transient" fails **open**: a future intent that installs
#     something would be ungated by omission, which is exactly the silence
#     `DEFAULT_REQUIRED` was written to avoid.
#
# Neither default is safe, so there is none. Every entry in `config.INTENTS`
# appears in exactly one set below, and a test enumerates `config.INTENTS` and
# fails on any intent in neither or in both. A new intent gets a red test, not
# a silent answer.

PERSISTS_AUTHORITY: frozenset[str] = frozenset({
    # Each installs something that runs later, under grants the installer no
    # longer has to be holding.
    "manage_monitor",     # event_bus._fire_action -> execute("code_executor")
    "manage_schedule",    # scheduler._async_run_handler, LOCAL_GRANTS
    "manage_procedure",   # procedure_executor.run_procedure, replayed on demand
    "manage_shortcut",    # a stored trigger that resolves to an intent later
    "manage_backup",      # enables a recurring off-machine upload
    # Mints the credential the browser extension presents. The credential
    # itself carries no intent authority -- the extension listener's ceiling is
    # empty, and nothing it sends can run an intent -- so this is not the
    # monitor-installs-a-callback shape. It is the other one the comment above
    # describes: a durable capability gain bought with a temporary grant.
    #
    # Before setup the browser tier drives a bundled Chromium, signed out of
    # everything. After it, and from then on, it drives the browser the user is
    # actually signed into. A half-hour raise could mint that token and leave
    # TENKA with permanent reach into the user's logged-in sessions long after
    # the raise expired. Standing EXECUTE at the keyboard, not a raise.
    "browser_extension_setup",
})

TRANSIENT_AUTHORITY: frozenset[str] = frozenset({
    # The effect ends with the turn. Running code is *not* durable in this
    # sense: `code_executor` can do anything a shell can inside the window,
    # and no in-process check changes that -- which is precisely what granting
    # EXECUTE means and why a raise is a deliberate act. What this pair stops
    # is TENKA's own machinery being used to make the window permanent.
    "small_talk", "unknown", "create_note", "open_browser", "get_time",
    "computer_task", "read_screen", "find_and_click", "code_executor",
    "memory_query", "start_recording", "stop_recording", "get_recording",
    "summarize_recording", "web_search", "browse_url", "file_task",
    "set_reminder", "cancel_reminder", "hide_avatar", "show_avatar",
    "meet_face", "recognize_face", "forget_face", "camera_look", "planner",
    "enroll_voice", "forget_voice", "store_memory", "browser_tabs",
    "forget_memory", "shutdown", "manifest_dispatch", "self_knowledge",
})
