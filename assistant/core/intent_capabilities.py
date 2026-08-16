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
    "browser_cdp_setup": Capability.EXECUTE,
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
    "store_memory": Capability.CHAT_SEND,
    "forget_memory": Capability.CHAT_SEND,
    "set_reminder": Capability.CHAT_SEND,
    "cancel_reminder": Capability.CHAT_SEND,
    "hide_avatar": Capability.CHAT_SEND,
    "show_avatar": Capability.CHAT_SEND,
}

# Unlisted intents require the strongest capability. See the module docstring.
DEFAULT_REQUIRED: Capability = Capability.EXECUTE
