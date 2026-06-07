"""Event source registry — sources self-register on import."""
from ...core.registry import RegistryBase
from .base import EventSource

source_registry: RegistryBase[EventSource] = RegistryBase("event_source")

from . import media, window  # noqa: E402, F401 — triggers registration
