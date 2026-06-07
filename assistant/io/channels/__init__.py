"""Messaging channel registry — adapters self-register on import."""
from ...core.registry import RegistryBase
from .base import Channel

channel_registry: RegistryBase[Channel] = RegistryBase("channel")
