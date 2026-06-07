"""
core/registry.py — Generic thread-safe key-to-object registry.

Zero project imports: this module lives in core/ and must stay
import-free of all other assistant packages so it can be used as
the foundation for tools, LLM providers, channels, and event sources.
"""

from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar

# ─── Types ────────────────────────────────────────────────────────────────────

T = TypeVar("T")


# ─── RegistryBase ─────────────────────────────────────────────────────────────

class RegistryBase(Generic[T]):
    """
    A thread-safe, generic registry that maps string keys to typed objects.

    Intended as the single shared primitive for all registries in the project
    (tool handlers, LLM providers, I/O channels, event-source factories, …).

    Args:
        name: Human-readable singular noun used in error messages,
              e.g. "tool", "provider", "channel".
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}
        self._lock = threading.Lock()

    # ─── Mutation ─────────────────────────────────────────────────────────────

    def register(self, key: str, obj: T) -> T:
        """
        Eagerly register *obj* under *key*.

        Returns *obj* so callers can chain: ``handler = registry.register("k", fn)``.
        Raises ``ValueError`` if *key* is already registered.
        """
        with self._lock:
            if key in self._entries:
                raise ValueError(
                    f"{self._name} '{key}' is already registered"
                )
            self._entries[key] = obj
        return obj

    def decorator(self, key: str) -> Callable[[T], T]:
        """
        Decorator form of :meth:`register`.

        Usage::

            @registry.decorator("my_key")
            def my_handler(...):
                ...
        """
        def _wrap(obj: T) -> T:
            self.register(key, obj)
            return obj
        return _wrap

    def reset(self) -> None:
        """Clear all entries. Intended for use in tests only."""
        with self._lock:
            self._entries.clear()

    # ─── Lookup ───────────────────────────────────────────────────────────────

    def get(self, key: str) -> T | None:
        """Return the object registered under *key*, or ``None`` if absent."""
        with self._lock:
            return self._entries.get(key)

    def require(self, key: str) -> T:
        """
        Return the object registered under *key*.

        Raises ``KeyError`` with a descriptive message if *key* is not found.
        """
        with self._lock:
            try:
                return self._entries[key]
            except KeyError:
                raise KeyError(
                    f"{self._name} registry has no entry '{key}'"
                ) from None

    def has(self, key: str) -> bool:
        """Return ``True`` if *key* is registered."""
        with self._lock:
            return key in self._entries

    def keys(self) -> list[str]:
        """Return a snapshot list of all registered keys."""
        with self._lock:
            return list(self._entries.keys())

    def list_all(self) -> dict[str, T]:
        """
        Return a caller-owned shallow copy of the registry.

        Mutations to the returned dict do not affect the registry.
        """
        with self._lock:
            return dict(self._entries)
