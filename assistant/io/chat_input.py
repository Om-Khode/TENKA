"""
chat_input.py — Terminal chat input with slash-command autocomplete.

Replaces the bare input() loop with prompt_toolkit so the user gets:
  - Dropdown completion after "/"
  - Setting descriptions as sidebar hints
  - Session-scoped command history (Up/Down arrows)
  - Safe co-existence with async logging (patch_stdout keeps prints above the prompt)

Falls back to plain input() if prompt_toolkit isn't installed, so the app still
starts for anyone on a stripped-down env.
"""

from __future__ import annotations

import logging
import time

from .. import config

_logger = logging.getLogger("chat_input")


def _build_completer():
    """Construct a Completer backed by the runtime settings registry.

    Returns None if prompt_toolkit is missing.
    """
    try:
        from prompt_toolkit.completion import Completer, Completion
    except ImportError:
        return None

    RESERVED_COMMANDS = [
        ("help", "show slash command list"),
        ("config", "list or inspect runtime settings"),
        ("set", "change a runtime setting"),
        ("reset", "revert a runtime setting to default"),
        ("compress", "compress conversation history now"),
    ]

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            # Everything after the leading "/"
            body = text[1:]

            # Tokenize what the user has typed so far
            parts = body.split()
            # `ends_with_space` — are we about to start a new token?
            ends_with_space = body.endswith(" ") or body == ""

            if not parts or (len(parts) == 1 and not ends_with_space):
                # User is typing the command name — offer reserved + setting keys
                prefix = parts[0] if parts else ""
                yield from self._command_completions(prefix)
                return

            cmd = parts[0].lower()

            # `/set <TAB>` or `/config <TAB>` or `/reset <TAB>` → setting keys
            if cmd in ("set", "config", "reset"):
                if ends_with_space and len(parts) == 1:
                    yield from self._setting_key_completions("")
                    return
                if len(parts) == 2 and not ends_with_space:
                    yield from self._setting_key_completions(parts[1])
                    return
                # `/set <key> <value>` — no completion on value
                return

            # Shortcut form `/<key> <value>` — no completion once past the key
            # (setting keys handled in the first branch)

        def _command_completions(self, prefix: str):
            from prompt_toolkit.completion import Completion
            prefix_lower = prefix.lower()

            # Reserved commands first
            for name, desc in RESERVED_COMMANDS:
                if name.startswith(prefix_lower):
                    yield Completion(
                        name,
                        start_position=-len(prefix),
                        display=name,
                        display_meta=desc,
                    )

            # Setting keys (shortcut form /<key>)
            for key in sorted(config.RUNTIME_SETTINGS_REGISTRY.keys()):
                if not key.startswith(prefix_lower):
                    continue
                meta = config.RUNTIME_SETTINGS_REGISTRY[key]
                hint = _short_description(meta)
                if meta.get("needs_restart"):
                    hint = f"[R] {hint}"
                yield Completion(
                    key,
                    start_position=-len(prefix),
                    display=key,
                    display_meta=hint,
                )

        def _setting_key_completions(self, prefix: str):
            from prompt_toolkit.completion import Completion
            prefix_lower = prefix.lower()
            for key in sorted(config.RUNTIME_SETTINGS_REGISTRY.keys()):
                if not key.startswith(prefix_lower):
                    continue
                meta = config.RUNTIME_SETTINGS_REGISTRY[key]
                cast_name = getattr(meta["cast"], "__name__", "?")
                hint = f"{cast_name} — {_short_description(meta)}"
                if meta.get("needs_restart"):
                    hint = f"[R] {hint}"
                yield Completion(
                    key,
                    start_position=-len(prefix),
                    display=key,
                    display_meta=hint,
                )

    return SlashCompleter()


def _short_description(meta: dict, max_len: int = 60) -> str:
    desc = (meta.get("description") or "").strip()
    if len(desc) > max_len:
        desc = desc[: max_len - 1] + "…"
    return desc or "(no description)"


def chat_input_loop(input_queue) -> None:
    """Background thread body: push lines typed by the user onto the queue.

    Tries prompt_toolkit first (autocomplete + history + stdout-safe prompt).
    Falls back to plain input() if the import fails or the terminal rejects
    the rich prompt (non-TTY, unsupported shell, etc.).
    """
    time.sleep(1.0)  # Let startup logs finish before first prompt
    session, patch_stdout = _try_build_session()

    if session is None:
        # Fallback: dumb input()
        while True:
            try:
                text = input("💬 Chat: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text:
                input_queue.put(("chat", text))
        return

    # prompt_toolkit path: wrap every prompt in patch_stdout so async log output
    # never overwrites the prompt line.
    while True:
        try:
            with patch_stdout():
                text = session.prompt("💬 Chat: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text:
            input_queue.put(("chat", text))


def _try_build_session():
    """Return (PromptSession, patch_stdout) or (None, None) on failure."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.patch_stdout import patch_stdout
    except ImportError:
        return None, None

    completer = _build_completer()
    try:
        session = PromptSession(
            completer=completer,
            history=InMemoryHistory(),
            complete_while_typing=True,
        )
    except Exception as e:
        _logger.debug(f"PromptSession init failed, falling back to plain input: {e}")
        return None, None
    return session, patch_stdout
