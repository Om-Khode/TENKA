"""
llm/providers/groq.py — Groq provider implementation.

Extracted from router.py. Covers chat, streaming, vision, and
synchronous vision yes/no checks.
"""

from __future__ import annotations

import itertools
import logging
import os

import requests

from ... import config
from .base import ProviderResult

logger = logging.getLogger("llm")


# ─── Key Rotation ────────────────────────────────────────────────────────────

def _load_groq_keys() -> list[str]:
    keys = []
    for i in range(1, 10):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)
    legacy = os.getenv("GROQ_API_KEY")
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys

GROQ_KEYS = _load_groq_keys()
_groq_key_cycle = itertools.cycle(GROQ_KEYS) if GROQ_KEYS else None
_current_groq_key = next(_groq_key_cycle) if _groq_key_cycle else None


# ─── GroqProvider ────────────────────────────────────────────────────────────

class GroqProvider:
    name = "groq"

    def is_available(self) -> bool:
        return bool(_current_groq_key or getattr(config, "GROQ_API_KEY", ""))

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
        messages: list[dict] | None = None,
    ) -> ProviderResult | None:
        global _current_groq_key
        global _groq_key_cycle
        target_model = model or getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")

        try:
            from groq import Groq

            if not _current_groq_key:
                api_key = getattr(config, "GROQ_API_KEY", "")
                key_index_str = "config"
            else:
                api_key = _current_groq_key
                key_index_str = f"#{GROQ_KEYS.index(api_key) + 1}"

            client = Groq(api_key=api_key, max_retries=0)

            if messages:
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(messages)
                all_messages.append({"role": "user", "content": user_message})
            else:
                all_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]

            try:
                response = client.chat.completions.create(
                    model=target_model,
                    messages=all_messages,
                    temperature=temperature if temperature is not None else 0.7,
                    max_tokens=max_tokens,
                    timeout=config.LLM_TIMEOUT,
                )
            except Exception as e:
                error_str = str(e).lower()
                if getattr(e, "status_code", None) == 429 or "429" in error_str or "rate" in error_str:
                    if GROQ_KEYS and len(GROQ_KEYS) > 1:
                        for _ in range(len(GROQ_KEYS) - 1):
                            _current_groq_key = next(_groq_key_cycle)
                            next_idx = GROQ_KEYS.index(_current_groq_key) + 1
                            logger.info(f"[LLM] Groq key {key_index_str} rate limited, rotating to key #{next_idx}")
                            try:
                                client = Groq(api_key=_current_groq_key, max_retries=0)
                                response = client.chat.completions.create(
                                    model=target_model,
                                    messages=all_messages,
                                    temperature=temperature if temperature is not None else 0.7,
                                    max_tokens=max_tokens,
                                    timeout=config.LLM_TIMEOUT,
                                )
                                logger.info(f"[LLM] Using Groq key #{next_idx} — response OK")
                                _tin = _tout = None
                                try:
                                    _tin = response.usage.prompt_tokens
                                    _tout = response.usage.completion_tokens
                                except Exception:
                                    pass
                                return ProviderResult(response.choices[0].message.content.strip(), _tin, _tout)
                            except Exception as retry_e:
                                if "429" in str(retry_e).lower():
                                    continue
                                raise retry_e
                        raise e
                    else:
                        raise e
                else:
                    raise e

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Groq returned None content — possible rate limit or empty response")
            text = content.strip()
            # Strip <think>...</think> blocks from reasoning models (e.g. Qwen3)
            if text.startswith("<think>"):
                import re as _re
                text = _re.sub(r'<think>.*?</think>\s*', '', text, flags=_re.DOTALL).strip()
            if not text:
                raise ValueError("Groq response empty after stripping think tags")
            logger.info(f'[LLM] Using Groq ({target_model}) — response: "{text[:100]}..."')
            _tin = _tout = None
            try:
                _tin = response.usage.prompt_tokens
                _tout = response.usage.completion_tokens
            except Exception:
                pass
            return ProviderResult(text, _tin, _tout)

        except ImportError:
            logger.warning("[LLM] groq package not installed — skipping Groq backend")
            return None
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                logger.warning(f"[LLM] Groq rate-limited: {e}")
            else:
                logger.error(f"[LLM] Groq error: {e}")
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
        """Async generator that yields token chunks from Groq."""
        global _current_groq_key
        target_model = model or getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")
        try:
            from groq import Groq
            import asyncio
            api_key = _current_groq_key
            if not api_key:
                api_key = getattr(config, "GROQ_API_KEY", "")
            if not api_key:
                return
            client = Groq(api_key=api_key, max_retries=0)
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
                    model=target_model,
                    messages=all_messages,
                    temperature=temperature if temperature is not None else 0.7,
                    max_tokens=max_tokens,
                    timeout=config.LLM_TIMEOUT,
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
                logger.info(f"[LLM] Streamed from Groq ({target_model})")
        except ImportError:
            logger.warning("[LLM] groq not installed — skipping Groq streaming")
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                logger.warning(f"[LLM] Groq stream rate-limited: {e}")
            else:
                logger.error(f"[LLM] Groq stream error: {e}")

    def vision(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
    ) -> ProviderResult | None:
        """Groq vision using llama-4-scout (the only free vision model on Groq)."""
        global _current_groq_key
        global _groq_key_cycle

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            from groq import Groq

            if not _current_groq_key:
                api_key = getattr(config, "GROQ_API_KEY", "")
            else:
                api_key = _current_groq_key

            client = Groq(api_key=api_key, max_retries=0)
            model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    timeout=config.LLM_TIMEOUT,
                )
            except Exception as e:
                error_str = str(e).lower()
                if getattr(e, "status_code", None) == 429 or "429" in error_str or "rate" in error_str:
                    if GROQ_KEYS and len(GROQ_KEYS) > 1:
                        _current_groq_key = next(_groq_key_cycle)
                        client = Groq(api_key=_current_groq_key, max_retries=0)
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=0,
                            max_tokens=2048,
                            timeout=config.LLM_TIMEOUT,
                        )
                    else:
                        raise e
                else:
                    raise e

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Groq returned None content")

            logger.info("[llm] INFO: Vision call — Groq llama-4-scout")
            _tin = _tout = None
            try:
                _tin = response.usage.prompt_tokens
                _tout = response.usage.completion_tokens
                tokens = response.usage.total_tokens
            except Exception:
                tokens = "unknown"
            logger.info(f"[llm] INFO: Vision response OK ({tokens} tokens)")
            return ProviderResult(content.strip(), _tin, _tout)

        except Exception as e:
            logger.warning(f"[llm] WARNING: Groq vision failed: {e}")
            return None

    def vision_yes_no_sync(self, image_base64: str, prompt: str) -> str:
        """
        Synchronous vision YES/NO check using Groq llama-4-scout.
        Used by the action executor (non-async context) to verify screen state.
        Returns "YES" or "NO". Returns "NO" on any error.
        """
        if not _current_groq_key:
            return "NO"
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {_current_groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "temperature": 0,
                    "max_tokens": 5,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                                },
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                },
                timeout=10,
            )
            result = response.json()["choices"][0]["message"]["content"].strip().upper()
            logger.info(f"[llm] _vision_yes_no_sync → {result}")
            return "YES" if "YES" in result else "NO"
        except Exception as e:
            logger.warning(f"[llm] _vision_yes_no_sync failed: {e}")
            return "NO"


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import provider_registry
    provider_registry.register("groq", GroqProvider())

_register()
