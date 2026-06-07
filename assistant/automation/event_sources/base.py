"""EventSource protocol for event-driven monitors."""
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class EventSource(Protocol):
    """Interface every event source must satisfy."""

    name: str
    event_types: frozenset[str]

    def start(self, dispatch_fn: Callable[[dict], None], **kwargs) -> None: ...
    def stop(self) -> None: ...
