"""
llm/providers/gemini.py — Google Gemini provider implementation.

Extracted from router.py. Covers chat, streaming, and vision.
"""

from __future__ import annotations

import base64
import logging

from ... import config
from .base import ProviderResult

logger = logging.getLogger("llm")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _close_genai_client(client) -> None:
    """Best-effort close of a google-genai Client to release the sync httpx
    pool and trigger the async sub-client cleanup deterministically rather
    than letting GC schedule an orphan aclose() task."""
    if client is None:
        return
    try:
        client.close()
    except Exception:
        pass
    aio = getattr(client, "aio", None)
    if aio is None:
        return
    closer = getattr(aio, "aclose", None)
    if closer is None:
        return
    try:
        import asyncio as _asyncio
        coro = closer()
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None or not loop.is_running():
            _asyncio.run(coro)
            return
        # Inside a running loop: schedule and track so it's not orphaned.
        task = loop.create_task(coro)
        task.add_done_callback(lambda _t: None)
    except Exception:
        pass


# ─── GeminiProvider ──────────────────────────────────────────────────────────

class GeminiProvider:
    name = "gemini"

    def is_available(self) -> bool:
        return bool(getattr(config, "GEMINI_API_KEY", ""))

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
        *,
        messages: list[dict] | None = None,
        thinking_budget: int | None = 0,
    ) -> ProviderResult | None:
        api_key = getattr(config, "GEMINI_API_KEY", "") or ""
        if not api_key:
            return None

        target_model = model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")

        client = None
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=config.LLM_TIMEOUT * 1000),
            )

            gen_config_kwargs: dict = {
                "temperature": temperature if temperature is not None else 0.7,
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                gen_config_kwargs["system_instruction"] = system_prompt
            if thinking_budget is not None and "flash-lite" not in target_model:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

            if messages:
                contents = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append(types.Content(
                        role=role, parts=[types.Part.from_text(text=msg["content"])]
                    ))
                contents.append(types.Content(
                    role="user", parts=[types.Part.from_text(text=user_message)]
                ))
            else:
                contents = user_message

            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )

            text = (response.text or "").strip()
            if text.startswith("<think>"):
                import re as _re
                text = _re.sub(r'<think>.*?</think>\s*', '', text, flags=_re.DOTALL).strip()
            if not text:
                raise ValueError("Gemini returned empty content")
            logger.info(f'[LLM] Using Gemini ({target_model}) — response: "{text[:100]}..."')
            _tin = _tout = None
            try:
                um = response.usage_metadata
                _tin = um.prompt_token_count
                _tout = um.candidates_token_count
            except Exception:
                pass
            return ProviderResult(text, _tin, _tout)

        except ImportError:
            logger.warning("[LLM] google-genai package not installed — pip install google-genai")
            return None
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                logger.warning(f"[LLM] Gemini rate-limited: {e}")
            else:
                logger.error(f"[LLM] Gemini error: {e}")
            return None
        finally:
            _close_genai_client(client)

    async def stream(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
        messages: list[dict] | None = None,
    ):
        """Async generator that yields token chunks from Gemini."""
        api_key = getattr(config, "GEMINI_API_KEY", "") or ""
        if not api_key:
            return
        target_model = model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
        client = None
        try:
            from google import genai
            from google.genai import types
            import asyncio
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=config.LLM_TIMEOUT * 1000),
            )
            gen_config_kwargs: dict = {
                "temperature": temperature if temperature is not None else 0.7,
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                gen_config_kwargs["system_instruction"] = system_prompt
            if "flash-lite" not in target_model:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            if messages:
                contents = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append(types.Content(
                        role=role, parts=[types.Part.from_text(text=msg["content"])]
                    ))
                contents.append(types.Content(
                    role="user", parts=[types.Part.from_text(text=user_message)]
                ))
            else:
                contents = user_message

            response_iter = await asyncio.to_thread(
                lambda: client.models.generate_content_stream(
                    model=target_model,
                    contents=contents,
                    config=types.GenerateContentConfig(**gen_config_kwargs),
                )
            )
            yielded = False
            for chunk in response_iter:
                text = chunk.text or ""
                if text:
                    yielded = True
                    yield text
            if yielded:
                logger.info(f"[LLM] Streamed from Gemini ({target_model})")
        except ImportError:
            logger.warning("[LLM] google-genai not installed — skipping Gemini streaming")
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str or "quota" in error_str:
                logger.warning(f"[LLM] Gemini stream rate-limited: {e}")
            else:
                logger.error(f"[LLM] Gemini stream error: {e}")
        finally:
            _close_genai_client(client)

    def vision(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> ProviderResult | None:
        api_key = getattr(config, "GEMINI_API_KEY", "") or ""
        if not api_key:
            return None

        target_model = model or getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")

        client = None
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=config.LLM_TIMEOUT * 1000),
            )

            image_bytes = base64.b64decode(image_base64)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            gen_config_kwargs: dict = {
                "temperature": 0,
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                gen_config_kwargs["system_instruction"] = system_prompt
            if "flash-lite" not in target_model:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            response = client.models.generate_content(
                model=target_model,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )

            text = (response.text or "").strip()
            if text.startswith("<think>"):
                import re as _re
                text = _re.sub(r'<think>.*?</think>\s*', '', text, flags=_re.DOTALL).strip()
            if not text:
                raise ValueError("Gemini vision returned empty content")

            _tin = _tout = None
            try:
                um = response.usage_metadata
                _tin = um.prompt_token_count
                _tout = um.candidates_token_count
                tokens = um.total_token_count
            except Exception:
                tokens = "unknown"
            logger.info(f"[llm] Vision (Gemini {target_model}) OK ({tokens} tokens)")
            return ProviderResult(text, _tin, _tout)

        except ImportError:
            logger.warning("[llm] google-genai not installed — skipping Gemini vision")
            return None
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                logger.warning(f"[llm] Gemini vision rate-limited: {e}")
            else:
                logger.error(f"[llm] Gemini vision error: {e}")
            return None
        finally:
            _close_genai_client(client)


# ─── Registration ────────────────────────────────────────────────────────────

def _register() -> None:
    from . import provider_registry
    provider_registry.register("gemini", GeminiProvider())

_register()
