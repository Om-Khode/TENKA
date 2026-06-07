"""manifest_primitives.py — primitive backends for manifest-based selector execution.

Currently implements hotkey + uia primitives. vision_reground is a stub
in v1 and gets filled in by healer.py in session 4 (it's the tier-2
healing path, not a primary primitive).

The `terminator` parameter is dependency-injected so tests use FakeTerminator
without touching the real Rust binding. The `except Exception:` clauses below
wrap external Terminator/PyO3 calls; we surface failure as a PrimitiveResult
rather than propagating so the dispatcher selector-chain walker can fall
through to the next selector cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .manifest_schema import Selector


@dataclass(frozen=True)
class PrimitiveResult:
    ok: bool
    error: str | None = None
    detail: dict[str, Any] | None = None  # reserved for healer (Session 4)


class TerminatorLike(Protocol):
    def send_key(self, key: str) -> None: ...
    def find_element(
        self,
        *,
        automation_id: str = "",
        name: str = "",
        control_type: str = "",
        window: str = "",
    ): ...
    def click(self, element: Any) -> None: ...


def execute_primitive(
    selector: Selector, *, terminator: TerminatorLike, active_window: str,
) -> PrimitiveResult:
    """Dispatch a Selector to its backend. manifest-based selector chain walker calls this."""
    if selector.kind == "hotkey":
        return _exec_hotkey(selector, terminator)
    if selector.kind == "uia":
        return _exec_uia(selector, terminator, active_window)
    if selector.kind == "vision_reground":
        return PrimitiveResult(ok=False, error="vision_reground not implemented as a primitive")
    return PrimitiveResult(ok=False, error=f"unknown selector.kind: {selector.kind}")


def _exec_hotkey(selector: Selector, terminator: TerminatorLike) -> PrimitiveResult:
    if not selector.keys:
        return PrimitiveResult(ok=False, error="hotkey selector missing 'keys'")
    try:
        terminator.send_key(selector.keys)
    except Exception as e:
        return PrimitiveResult(ok=False, error=f"send_key failed: {e}")
    return PrimitiveResult(ok=True)


def _exec_uia(
    selector: Selector, terminator: TerminatorLike, active_window: str,
) -> PrimitiveResult:
    # UIA selectors identify by automation_id (most stable) OR by the
    # accessible name (name_hint). The schema (manifest_schema.Selector)
    # has always allowed either — apps with no automation_id exposure
    # (Spotify, most Electron apps, web players) cannot supply one.
    if not selector.automation_id and not selector.name_hint:
        return PrimitiveResult(
            ok=False, error="uia selector missing both 'automation_id' and 'name_hint'",
        )
    try:
        elem = terminator.find_element(
            automation_id=selector.automation_id or "",
            name=selector.name_hint or "",
            control_type=selector.control_type or "",
            window=active_window,
        )
    except Exception as e:
        return PrimitiveResult(ok=False, error=f"find_element failed: {e}")
    try:
        terminator.click(elem)
    except Exception as e:
        return PrimitiveResult(ok=False, error=f"click failed: {e}")
    return PrimitiveResult(ok=True)
