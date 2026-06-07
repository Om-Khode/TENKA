# assistant/io/overlay/ipc.py
"""overlay IPC parser.

Parses JSON-line events from the engine. See spec §8.

Pure function, no I/O. Used by overlay/__main__.py to validate stdin lines.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .theme import PHASE_STYLES

_PROTOCOL_CURRENT = 2
_PROTOCOL_SUPPORTED = (1, 2)  # accept v1 with defaulted step/tier for forward-compat


@dataclass
class ParseResult:
    ok: bool
    event: dict | None = None
    error: str | None = None
    warning: str | None = None


def parse_event(line: str) -> ParseResult:
    line = line.strip()
    if not line:
        return ParseResult(ok=False, error="empty line")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as e:
        return ParseResult(ok=False, error=f"invalid json: {e}")

    if not isinstance(payload, dict):
        return ParseResult(ok=False, error="payload not an object")

    if "v" not in payload:
        return ParseResult(ok=False, error="missing 'v' field")
    if payload["v"] not in _PROTOCOL_SUPPORTED:
        return ParseResult(
            ok=False,
            error=f"version mismatch: got {payload['v']}, supported {_PROTOCOL_SUPPORTED}",
        )

    if "type" not in payload:
        return ParseResult(ok=False, error="missing 'type' field")

    if payload["type"] == "status":
        if "phase" not in payload:
            return ParseResult(ok=False, error="status event missing 'phase'")
        payload.setdefault("detail", "")
        payload.setdefault("cursor_follows", False)
        payload.setdefault("ts", 0.0)
        payload.setdefault("step", None)  # v1 events get None
        payload.setdefault("tier", None)
        warning = None
        if payload["phase"] not in PHASE_STYLES:
            warning = f"unknown phase '{payload['phase']}' — falling back to default"
        return ParseResult(ok=True, event=payload, warning=warning)

    if payload["type"] == "cmd":
        if "cmd" not in payload:
            return ParseResult(ok=False, error="cmd event missing 'cmd'")
        return ParseResult(ok=True, event=payload)

    return ParseResult(ok=False, error=f"unknown type '{payload['type']}'")
