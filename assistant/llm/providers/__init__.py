"""
llm/providers/ — LLM provider registry.

Importing this package creates the singleton registry and triggers
registration of all built-in providers.
"""

from ...core.registry import RegistryBase
from .base import Provider

provider_registry: RegistryBase[Provider] = RegistryBase("provider")

# Trigger self-registration in each provider module
from . import gemini, groq, cerebras, ollama  # noqa: E402, F401
