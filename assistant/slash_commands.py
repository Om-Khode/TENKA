"""
slash_commands.py — Runtime Config Slash Commands (RC-1c)

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


RESERVED = {"help", "config", "set", "reset", "compress", "promote"}


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
