"""Channel protocol for messaging adapters."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Channel(Protocol):
    """Interface every messaging adapter must satisfy."""

    name: str

    async def send(self, message: str, recipient: str | None = None) -> bool: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def execute(self, action: str, params: dict) -> dict | list | str: ...
