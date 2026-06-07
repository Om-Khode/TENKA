"""
llm/providers/base.py — Provider protocol and shared result type.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable


# ─── Result Type ─────────────────────────────────────────────────────────────

class ProviderResult(NamedTuple):
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None


# ─── Provider Protocol ───────────────────────────────────────────────────────

@runtime_checkable
class Provider(Protocol):
    name: str

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResult | None: ...

    def is_available(self) -> bool: ...
