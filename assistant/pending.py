"""Pending state — multi-turn dialog tracker.

A PendingState owns one interactive dialog state (destructive op confirmation,
oauth setup, messaging disambiguation, teaching session, etc.). It carries a
payload, a touch timestamp, and a timeout; reading the payload after the
timeout has elapsed auto-clears the state.

Eventually relocates to `actions/pending/base.py` once `actions.py` is split
into a package.

Usage:

    from assistant.pending import PendingState, pending_registry

    destructive = pending_registry.register(
        PendingState[dict]("destructive", timeout=30.0)
    )

    # start a pending interaction
    destructive.set({"op": "delete", "path": some_path})

    # later, in the handler
    payload = destructive.payload   # None if cleared or expired
    if payload is None:
        return None                 # not in a destructive flow
    payload["dest_folder"] = candidate
    destructive.touch()             # reset the timeout
"""
import time
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class PendingState(Generic[T]):
    """Single multi-turn interaction state."""

    def __init__(self, name: str, timeout: float):
        self.name = name
        self.timeout = timeout
        self._payload: Optional[T] = None
        self._ts: float = 0.0

    def set(self, payload: T) -> None:
        """Start (or replace) the pending state."""
        self._payload = payload
        self._ts = time.time()

    def touch(self) -> None:
        """Reset the timeout without changing the payload (for re-prompts)."""
        if self._payload is not None:
            self._ts = time.time()

    def clear(self) -> None:
        """End the pending state."""
        self._payload = None
        self._ts = 0.0

    @property
    def payload(self) -> Optional[T]:
        """Current payload, or None if inactive/expired.

        Reading an expired state clears it as a side effect.
        """
        if self._payload is None:
            return None
        if time.time() - self._ts > self.timeout:
            self.clear()
            return None
        return self._payload

    @property
    def active(self) -> bool:
        """True iff payload is set and not expired. Clears on expiry."""
        return self.payload is not None

    @property
    def age(self) -> float:
        """Seconds since the state was last touched. 0.0 if inactive."""
        if self._payload is None:
            return 0.0
        return time.time() - self._ts


class PendingRegistry:
    """Registry of all PendingStates so the planner can snapshot them.

    Replaces the reflection-based `_PENDING_VARS` list of string identifiers.
    """

    def __init__(self):
        self._states: dict[str, PendingState] = {}

    def register(self, state: PendingState) -> PendingState:
        """Register a state under its `name`. Returns the state for chaining.

        Raises ValueError if a state with the same name is already registered.
        """
        if state.name in self._states:
            raise ValueError(f"PendingState '{state.name}' already registered")
        self._states[state.name] = state
        return state

    def get(self, name: str) -> Optional[PendingState]:
        return self._states.get(name)

    def snapshot(self) -> dict[str, bool]:
        """Return {name: active} for every registered state.

        The planner calls this before and after a step to detect whether the
        step triggered a user-interaction flow (a state that flipped from
        inactive to active).
        """
        return {name: s.active for name, s in self._states.items()}

    def any_active(self, exclude: set[str] | None = None) -> bool:
        """True if any registered state is active (optionally excluding some)."""
        for name, state in self._states.items():
            if exclude and name in exclude:
                continue
            if state.active:
                return True
        return False

    def names(self) -> list[str]:
        return list(self._states.keys())


# Module-level singleton — the shared registry every PendingState joins.
pending_registry = PendingRegistry()
