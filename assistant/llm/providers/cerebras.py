"""
llm/providers/cerebras.py — Cerebras provider implementation.

Extracted from router.py. Covers chat and streaming.
"""

from __future__ import annotations

import logging

from ... import config
from .base import ProviderResult

logger = logging.getLogger("llm")


# ─── CerebrasProvider ───────────────────────────────────────────────────────

class CerebrasProvider:
    name = "cerebras"

    def is_available(self) -> bool:
        return bool(getattr(config, "CEREBRAS_API_KEY", ""))

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
        messages: list[dict] | None = None,
    ) -> ProviderResult | None:
        try:
            from cerebras.cloud.sdk import Cerebras

            api_key = getattr(config, "CEREBRAS_API_KEY", "") or ""
            if not api_key:
                return None

            client = Cerebras(api_key=api_key)

            model_name = model or getattr(config, "CEREBRAS_MODEL", "gpt-oss-120b")

            if messages:
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(messages)
                all_messages.append({"role": "user", "content": user_message})
            else:
                all_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]

            response = client.chat.completions.create(
                model=model_name,
                messages=all_messages,
                temperature=temperature if temperature is not None else 0.7,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Cerebras returned None content — possible rate limit or empty response")
            text = content.strip()
            logger.info(f'[LLM] Falling back to Cerebras — response: "{text[:100]}..."')
            _tin = _tout = None
            try:
                _tin = response.usage.prompt_tokens
                _tout = response.usage.completion_tokens
            except Exception:
                pass
            return ProviderResult(text, _tin, _tout)

        except ImportError:
            logger.warning("[LLM] cerebras-cloud-sdk not installed — pip install cerebras-cloud-sdk")
            return None
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str or "quota" in error_str:
                logger.warning(f"[LLM] Cerebras rate-limited: {e}")
            else:
                logger.error(f"[LLM] Cerebras error: {e}")
            return None

    async def stream(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
        messages: list[dict] | None = None,
    ):
        """Async generator that yields token chunks from Cerebras."""
        try:
            from cerebras.cloud.sdk import Cerebras
            import asyncio
            api_key = getattr(config, "CEREBRAS_API_KEY", "") or ""
            if not api_key:
                return
            client = Cerebras(api_key=api_key)
            model_name = model or getattr(config, "CEREBRAS_MODEL", "gpt-oss-120b")
            if messages:
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(messages)
                all_messages.append({"role": "user", "content": user_message})
            else:
                all_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            stream = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=model_name,
                    messages=all_messages,
                    temperature=temperature if temperature is not None else 0.7,
                    max_tokens=max_tokens,
                    stream=True,
                )
            )
            yielded = False
            for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    yielded = True
                    yield content
            if yielded:
                logger.info(f"[LLM] Streamed from Cerebras ({model_name})")
        except ImportError:
            logger.warning("[LLM] cerebras-cloud-sdk not installed — skipping Cerebras streaming")
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                logger.warning(f"[LLM] Cerebras stream rate-limited: {e}")
            else:
                logger.error(f"[LLM] Cerebras stream error: {e}")


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import provider_registry
    provider_registry.register("cerebras", CerebrasProvider())

_register()
