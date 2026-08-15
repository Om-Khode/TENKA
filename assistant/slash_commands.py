"""
slash_commands.py — Runtime Config Slash Commands

Zero-LLM-cost command parser for user-facing runtime config. Recognized at
the very top of the text pipeline, before teaching/shortcuts/intent.

Grammar:
    /help                          → show command list
    /config                        → list every registered setting + value
    /config <key>                  → show one setting's value + description
    /set <key> <value>             → write a value (persists across sessions)
    /reset <key>                   → revert a setting to its default
    /<key>                         → shortcut for /config <key>
    /<key> <value>                 → shortcut for /set <key> <value>

Reserved command names (cannot collide with setting keys): help, config, set, reset.
Every other /word is treated as a setting shortcut.
"""

from typing import Optional

from . import config, settings


RESERVED = {"help", "config", "set", "reset", "compress", "promote", "studio"}


# ─── Background task strong-refs ───────────────────────────────────────────
# asyncio's _all_tasks is a WeakSet, so a fire-and-forget create_task() can
# be garbage-collected before it completes, silently killing the cycle with
# no log entry. Holding our own strong references in this set — and using
# the standard discard-on-done callback — prevents that.
_background_tasks: set = set()


HELP_TEXT = """Runtime config commands:
  /config                    — list all settings
  /config <key>              — show one setting's value + description
  /set <key> <value>         — set a value (persists across sessions)
  /reset <key>               — revert to default
  /compress                  — compress conversation history
  /promote                   — trigger a manifest-based promotion cycle (background)
  /studio devices            — list Studio device tokens
  /studio pair [label]       — mint a pairing code + QR for a new device
  /studio revoke <device_id> — revoke one Studio device
  /studio revoke all confirm — revoke every Studio device (re-pairing required)
  /help                      — this message

Shortcuts:
  /<key> <value>             — same as /set <key> <value>
  /<key>                     — same as /config <key>"""


def is_slash_command(text: str) -> bool:
    """True if `text` should be routed to handle() instead of intent detection."""
    stripped = text.lstrip()
    return stripped.startswith("/") and len(stripped) > 1


def handle(text: str) -> str:
    """Parse and execute a slash command. Always returns a non-empty response string."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "Not a slash command."

    # Split into at most 3 parts: cmd, key_or_value, rest
    parts = stripped[1:].split(maxsplit=2)
    if not parts:
        return HELP_TEXT

    cmd = parts[0].lower()

    if cmd == "help":
        return HELP_TEXT

    if cmd == "config":
        if len(parts) == 1:
            return _format_all_settings()
        return _format_one_setting(parts[1])

    if cmd == "set":
        if len(parts) < 3:
            return "Usage: /set <key> <value>"
        return _set_setting(parts[1], parts[2])

    if cmd == "reset":
        if len(parts) < 2:
            return "Usage: /reset <key>"
        return _reset_setting(parts[1])

    if cmd == "compress":
        return _compress_context()

    if cmd == "promote":
        return _promote_me1()

    if cmd == "studio":
        return _studio_command(parts[1:])

    # Shortcut form: /<key> [value]
    key = cmd
    if key in RESERVED:
        # Shouldn't reach here — RESERVED names match the explicit branches above
        return HELP_TEXT
    if key not in config.RUNTIME_SETTINGS_REGISTRY:
        return (
            f"Unknown setting: {key}. "
            f"Try /config to list available settings."
        )
    if len(parts) == 1:
        return _format_one_setting(key)
    # Join parts[1..] so values with spaces survive (e.g. /response_verbosity very detailed)
    raw_value = " ".join(parts[1:])
    return _set_setting(key, raw_value)


# ─── Formatters ──────────────────────────────────────────────────────────────


def _format_all_settings() -> str:
    stored = settings.list_all()
    if not config.RUNTIME_SETTINGS_REGISTRY:
        return "No runtime settings registered."

    lines = ["Runtime settings  (* = customized, R = needs restart):"]
    for key in sorted(config.RUNTIME_SETTINGS_REGISTRY.keys()):
        meta = config.RUNTIME_SETTINGS_REGISTRY[key]
        if key in stored:
            value = stored[key]
            custom_marker = "*"
        else:
            value = meta["default"]
            custom_marker = " "
        restart_marker = "R" if meta.get("needs_restart") else " "
        lines.append(f"  {custom_marker}{restart_marker} {key} = {value!r}")
    lines.append("")
    lines.append("Use /config <key> for description, /set <key> <value> to change.")
    return "\n".join(lines)


def _format_one_setting(key: str) -> str:
    meta = config.RUNTIME_SETTINGS_REGISTRY.get(key)
    if not meta:
        return f"Unknown setting: {key}"

    stored = settings.get(key)
    is_custom = stored is not None
    current = stored if is_custom else meta["default"]
    cast_name = getattr(meta["cast"], "__name__", str(meta["cast"]))
    description = meta.get("description") or "(no description)"
    restart_note = (
        "\n  note: changes require a restart to take effect"
        if meta.get("needs_restart") else ""
    )

    return (
        f"{key} = {current!r}\n"
        f"  default: {meta['default']!r}\n"
        f"  type: {cast_name}\n"
        f"  source: {'custom' if is_custom else 'default'}\n"
        f"  description: {description}"
        f"{restart_note}"
    )


# ─── Context Compression ────────────────────────────────────────────────────


def _compress_context() -> str:
    """Clear compression cache, forcing re-compression on next turn."""
    try:
        import assistant.main as main_mod
        main_mod._compression_cache = None
        return "Conversation compressed. Fresh summary will be generated on next message."
    except Exception:
        return "Conversation compressed."


# ─── manifest-based Promotion Trigger ─────────────────────────────────────────────────


def _promote_me1() -> str:
    """Schedule a manifest-based promotion cycle on the running event loop.

    Returns immediately. The cycle runs as a background task; its summary
    is logged at INFO level when complete. Never blocks the caller.
    """
    import asyncio
    import logging

    from .automation import promoter as promoter_mod
    from .automation.manifest_registry import get_singleton
    from .automation.promoter import Promoter
    from .storage.db import get_db
    from .storage.repos.automation_cache import AutomationCacheRepo

    logger = logging.getLogger("manifest")

    registry = get_singleton()
    db = get_db()
    if registry is None or db is None:
        return "manifest-based not initialized."

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — slash_commands.handle() is normally called from
        # inside an async function, but stay defensive for non-standard
        # callers (tests, ad-hoc CLI).
        return "Cannot schedule: no async loop available."

    # Debounce concurrent cycles — the data side is idempotent, but two
    # parallel run_once() calls would each hit find_unpromoted() before
    # either marked rows promoted, duplicating LLM spend.
    if promoter_mod.is_promotion_in_flight():
        return "manifest-based promotion already in progress."

    def _log_result(task: "asyncio.Task") -> None:
        promoter_mod._set_in_flight(False)
        if task.cancelled():
            logger.warning("[manifest promote] /promote task was cancelled")
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                f"[manifest promote] /promote task failed: {exc}", exc_info=exc,
            )
            return
        logger.info(f"[manifest promote] /promote summary: {task.result()}")

    promoter_mod._set_in_flight(True)
    try:
        promoter = Promoter(
            automation_cache_repo=AutomationCacheRepo(db),
            manifest_store=registry.store,
        )
        task = loop.create_task(promoter.run_once())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_log_result)
    except Exception as e:
        # Release the debounce on the failure path — _log_result never runs
        # if the task was never scheduled.
        promoter_mod._set_in_flight(False)
        logger.warning(
            f"[manifest promote] /promote scheduling failed: {e}", exc_info=True,
        )
        return f"manifest-based promotion scheduling failed: {e}"
    return "manifest-based promotion cycle scheduled. Results will be logged."


# ─── Studio device revocation ──────────────────────────────────────────────
# Local-only: revocation deliberately has no HTTP route. A route behind
# `system_control` would be reachable by the very token being revoked
# (Milestone 5 issues one token holding every grant), and pairing's trust
# anchor -- "you are physically at the desktop" -- should be the same
# anchor revocation rests on. See the Milestone 5/6 design docs for the
# full reasoning. Per-device revocation from a *paired phone's* own view
# belongs to the future pairing UI (Milestone 6), not here.

_STUDIO_USAGE = (
    "Usage:\n"
    "  /studio devices              — list issued Studio device tokens\n"
    "  /studio pair [label]         — mint a pairing code + QR for a new device\n"
    "  /studio revoke <device_id>   — revoke one device\n"
    "  /studio revoke all confirm   — revoke every device (rotates the "
    "instance secret; every paired device must be re-paired)"
)

# ─── Studio pairing ─────────────────────────────────────────────────────────
# Local-only for a different reason than revocation above: the store this
# reaches is `assistant.main._studio_pair_store`, not a file. See that
# global's docstring in main.py for why a fresh `PairCodeStore()` here would
# be a trap -- it would mint a code `POST /v1/pair` could never redeem.

_STUDIO_PAIR_LABEL_DEFAULT = "device"


def _studio_pair_default_grants():
    """What a code mints by default, absent any way for this command to
    name a device to intersect against.

    `POST /v1/pair/code` narrows a requested grant set to the *minting
    device's own* grants, so a device holding only SYSTEM_CONTROL can never
    mint a wider code carrying RECALL/FILES/CHAT_SEND (see
    routes/pairing.py). A slash command has no device to intersect against
    -- the restraint has to take a different shape here. Rather than
    defaulting to "everything" (the bootstrap admin token's own choice,
    made once, for the one device that starts the daemon), this refuses
    the single most consequential capability by default: SYSTEM_CONTROL
    reaches shutdown and backup control (manage_backup), not just chat and
    observation. Every other capability rides along, so a phone paired
    this way is still a fully useful remote control -- it just cannot
    administer the machine unless someone deliberately widens it later.

    A function, not a module-level constant: `Capability` is an `io/`
    type, and this file follows the codebase's deferred-import convention
    for that package (see `_studio_vault()` below) rather than importing it
    at module top.
    """
    from .io.api.vault import Capability
    return frozenset(Capability) - {Capability.SYSTEM_CONTROL}


def _studio_vault():
    """A fresh TokenVault over the real vault root.

    Deferred import: `io/` follows the codebase convention of importing
    inside the function, not at module top (see `from ..io.audio import
    tts`). Constructing a new instance per call -- rather than reaching
    into `assistant.main`'s running daemon -- means `/studio` works
    whether or not the Studio daemon happens to be running right now, and
    keeps this command testable against an isolated vault root without
    touching the live assistant process.
    """
    from .io.api.vault import TokenVault
    return TokenVault(config.SANDBOX_DIR)


def _studio_command(args: list) -> str:
    if not args:
        return _STUDIO_USAGE
    sub = args[0].lower()
    if sub == "devices":
        return _studio_list_devices()
    if sub == "pair":
        label = " ".join(args[1:]).strip() or _STUDIO_PAIR_LABEL_DEFAULT
        return _studio_pair(label)
    if sub == "revoke":
        return _studio_revoke(args[1] if len(args) > 1 else "")
    return _STUDIO_USAGE


def _studio_list_devices() -> str:
    vault = _studio_vault()
    devices = vault.devices()
    if not devices:
        return "No Studio devices issued."

    lines = ["Studio devices:"]
    for device in sorted(devices, key=lambda d: d.created_at):
        grants = ", ".join(sorted(g.value for g in device.grants))
        # `last_seen_at` is `None` for a device that authenticated at
        # issue time only and never made a second request -- printing the
        # literal "None" would read as a bug; "never" reads as a fact.
        last_seen = device.last_seen_at if device.last_seen_at else "never"
        lines.append(
            f"  {device.device_id}  {device.label!r}  [{grants}]  "
            f"created {device.created_at}  last seen {last_seen}"
        )
    return "\n".join(lines)


def _studio_pair(label: str) -> str:
    """Mint a pair code + QR into the *running daemon's* store.

    Reaches `assistant.main._studio_pair_store` rather than constructing a
    `PairCodeStore()` of its own -- see that global's docstring for why the
    latter would print a code `POST /v1/pair` could never redeem. When the
    daemon has never started (or isn't running), there is no store at all,
    and a code that cannot be redeemed is worse than an honest refusal.
    """
    import assistant.main as main_mod

    store = main_mod._studio_pair_store
    if store is None:
        return ("The Studio daemon is not running, so there is no pairing "
                "store to mint a code into. Start TENKA with the daemon "
                "enabled, then try /studio pair again.")

    try:
        pair_code = store.mint(label, _studio_pair_default_grants())
    except ValueError:
        # Unreachable today -- _studio_pair_default_grants() is never
        # empty -- but mint() raises ValueError on an empty grant set, and
        # that must never surface as a raw traceback on the console.
        return "Could not mint a pair code: no capabilities to grant."

    import time as _time
    from datetime import datetime, timedelta, timezone

    remaining = max(0.0, pair_code.expires_at - _time.monotonic())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=remaining)

    # Same payload shape `POST /v1/pair/code` encodes into its SVG
    # (`endpoint/pair#code`) -- a phone scanning either QR lands on the
    # same pairing URL, with the code in the fragment so it is never sent
    # to a server, proxy, or tunnel in between.
    endpoint = f"http://127.0.0.1:{config.STUDIO_API_PORT}"
    ascii_qr = _pair_code_ascii_qr(f"{endpoint}/pair#{pair_code.code}")

    # The code is deliberately NOT on the first line. main.py speaks (and,
    # before this task, logged) only `response.split("\n", 1)[0]` for every
    # non-"chat" source -- so a code sitting on line one would ride straight
    # into `tts.speak()`'s log line the moment this command is ever reached
    # by voice. `redact_secrets()` cannot be trusted to catch it either: at
    # 9 characters it clears none of that function's thresholds (the bare
    # path wants >=24, the labelled path wants a recognised role noun
    # immediately before the value, and "Pair code for 'x':" is neither).
    # Keeping the code off line one means the one line TTS ever sees never
    # contains it, regardless of source -- structurally, not by relying on
    # a heuristic to strip it back out after the fact.
    return (
        f"Pair code minted for {label!r} -- see below. "
        f"Expires at {expires_at.isoformat()} (~{int(remaining)}s from now).\n"
        f"Code: {pair_code.code}\n"
        f"Scan at {endpoint}/pair or enter the code manually.\n"
        f"{ascii_qr}"
    )


def _pair_code_ascii_qr(payload: str) -> str:
    """Render `payload` as a block-character QR code a terminal can show.

    `qr_svg()` (io/api/qr.py) renders `<svg>...<svg>`, which is exactly
    right for a browser dialog and useless in a console. `qrcode`'s own
    `print_ascii()` writes straight to a stream rather than returning a
    string, so it is rendered into an `io.StringIO` and read back.
    """
    import io as _io

    import qrcode

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(payload)
    qr.make(fit=True)
    buffer = _io.StringIO()
    qr.print_ascii(out=buffer)
    return buffer.getvalue()


def _studio_revoke(raw_target: str) -> str:
    vault = _studio_vault()
    tokens = raw_target.split()
    if not tokens:
        return _STUDIO_USAGE

    if tokens[0].lower() == "all":
        if len(tokens) == 1:
            return (
                "Revoking all devices rotates the instance secret and is "
                "irreversible: every paired device (phone, browser, anything "
                "holding a Studio token) stops working immediately and must "
                "be re-paired from scratch. To proceed: /studio revoke all confirm"
            )
        if len(tokens) == 2 and tokens[1].lower() == "confirm":
            vault.reset()
            return (
                "All Studio devices revoked -- instance secret rotated. "
                "Every previously issued token is now invalid; re-pair each device."
            )
        return _STUDIO_USAGE

    if len(tokens) > 1:
        return _STUDIO_USAGE

    device_id = tokens[0]
    from .io.api.vault import VaultUnavailableError
    try:
        revoked = vault.revoke(device_id)
    except VaultUnavailableError:
        # Either half of a lock: devices.json could not be read (so whether
        # device_id exists is genuinely unknown) or could not be written back
        # (so the revoke did not happen even though the read that preceded it
        # may have succeeded). Reporting "nothing was revoked" for either
        # would be a guess dressed up as an answer, on the one command that
        # has to tell the truth about whether a device is actually gone.
        return ("Could not read or write the Studio device list right now -- "
                "something else may have it open. Try again in a moment.")
    if revoked:
        return f"Revoked Studio device {device_id}."
    return f"No Studio device found with id {device_id!r} -- nothing was revoked."


# ─── Mutators ────────────────────────────────────────────────────────────────


def _coerce(raw_value: str, cast) -> "tuple[bool, object]":
    """Return (ok, value). Bool is parsed from common truthy/falsy tokens."""
    try:
        if cast is bool:
            token = raw_value.strip().lower()
            if token in ("true", "1", "yes", "on"):
                return True, True
            if token in ("false", "0", "no", "off"):
                return True, False
            return False, None
        return True, cast(raw_value)
    except (ValueError, TypeError):
        return False, None


def _set_setting(key: str, raw_value: str) -> str:
    meta = config.RUNTIME_SETTINGS_REGISTRY.get(key)
    if not meta:
        return f"Unknown setting: {key}. Try /config to list available settings."

    ok, value = _coerce(raw_value, meta["cast"])
    if not ok:
        cast_name = getattr(meta["cast"], "__name__", str(meta["cast"]))
        return f"Invalid value for {key}: expected {cast_name}, got {raw_value!r}"

    # Special handling for personality switching
    if key == "personality":
        from assistant import personality as _pers
        result = _pers.switch_personality(str(value))
        if result.startswith("Unknown"):
            return result
        try:
            import assistant.main as main_mod
            main_mod._compression_cache = None
        except Exception:
            pass
        settings.set(key, value, source="user")
        return result

    settings.set(key, value, source="user")
    config.reload_runtime_settings()
    suffix = (
        " Restart required for this to take effect."
        if meta.get("needs_restart") else ""
    )
    return f"Set {key} = {value!r}. Persists across sessions.{suffix}"


def _reset_setting(key: str) -> str:
    meta = config.RUNTIME_SETTINGS_REGISTRY.get(key)
    if not meta:
        return f"Unknown setting: {key}"

    existed = settings.delete(key)
    config.reload_runtime_settings()
    if existed:
        return f"Reset {key} to default ({meta['default']!r})."
    return f"{key} was already at default ({meta['default']!r})."
