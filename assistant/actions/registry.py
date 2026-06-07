"""Tool registry — single source of truth for intent → handler dispatch.

Handlers self-register via ``@tool_registry.decorator("intent_name")``
in their respective modules. The ``actions/__init__.py`` module imports
those modules to trigger registration, then dispatches via
``tool_registry.get(intent)``.
"""
from typing import Any, Awaitable, Callable

from ..core.registry import RegistryBase

Handler = Callable[[dict, str, Any], Awaitable[str]]

tool_registry: RegistryBase[Handler] = RegistryBase("tool")
