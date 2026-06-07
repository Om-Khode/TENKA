"""
llm/providers/ollama.py — Ollama (local) provider implementation.

Extracted from router.py. Covers chat and streaming.
"""

from __future__ import annotations

import json
import logging

import requests

from ... import config
from .base import ProviderResult

logger = logging.getLogger("llm")


# ─── OllamaProvider ─────────────────────────────────────────────────────────

class OllamaProvider:
    name = "ollama"

    def is_available(self) -> bool:
        # Ollama has no API key — availability depends on the server being up.
        # Return True optimistically; the actual call handles ConnectionError.
        return True

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
        messages: list[dict] | None = None,
    ) -> ProviderResult | None:
        url = config.OLLAMA_URL.rstrip("/") + "/api/chat"

        if messages:
            all_messages = [{"role": "system", "content": system_prompt}]
            all_messages.extend(messages)
            all_messages.append({"role": "user", "content": user_message})
        else:
            all_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        payload = {
            "model": model or config.OLLAMA_MODEL,
            "messages": all_messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else 0.7,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=config.LLM_TIMEOUT)
            resp.raise_for_status()

            data = resp.json()
            text = data.get("message", {}).get("content", "").strip()
            logger.info(f'[LLM] Falling back to Ollama — response: "{text[:100]}..."')
            if not text:
                return None
            return ProviderResult(
                text,
                data.get("prompt_eval_count"),
                data.get("eval_count"),
            )

        except requests.ConnectionError:
            logger.error(
                f"[LLM] Cannot connect to Ollama at {config.OLLAMA_URL}. "
                "Make sure Ollama is running (ollama serve)."
            )
            return None
        except Exception as e:
            logger.error(f"[LLM] Ollama error: {e}")
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
        """Async generator that yields token chunks from Ollama."""
        url = config.OLLAMA_URL.rstrip("/") + "/api/chat"
        if messages:
            all_messages = [{"role": "system", "content": system_prompt}]
            all_messages.extend(messages)
            all_messages.append({"role": "user", "content": user_message})
        else:
            all_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        payload = {
            "model": model or config.OLLAMA_MODEL,
            "messages": all_messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else 0.7,
                "num_predict": max_tokens,
            },
        }
        try:
            import asyncio
            response = await asyncio.to_thread(
                lambda: requests.post(url, json=payload, timeout=config.LLM_TIMEOUT, stream=True)
            )
            response.raise_for_status()
            yielded = False
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yielded = True
                    yield content
                if data.get("done"):
                    break
            if yielded:
                logger.info("[LLM] Streamed from Ollama")
        except requests.ConnectionError:
            logger.error(f"[LLM] Cannot connect to Ollama at {config.OLLAMA_URL} for streaming")
        except Exception as e:
            logger.error(f"[LLM] Ollama stream error: {e}")


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import provider_registry
    provider_registry.register("ollama", OllamaProvider())

_register()
