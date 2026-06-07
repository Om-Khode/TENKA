"""
llm.py — LLM module for the TENKA Voice Assistant.

Four-provider fallback chain:
  1. Primary:  Google Gemini (unified multimodal — Flash + Flash-Lite)
  2. Fallback: Groq cloud API (llama-3.3-70b, kimi-k2, etc.)
  3. Fallback: Cerebras cloud API (gpt-oss-120b — synthesis workhorse)
  4. Local:    Ollama server (llama3.1:8b)

The module tries each provider in order. If one fails (no API key,
network error, rate limit, etc.), it falls back to the next.

Provider implementations live in llm/providers/. This module dispatches
via the provider_registry singleton.
"""

import json
import logging
import time as _time_mod
from dataclasses import dataclass

from .. import config
from .providers import provider_registry
from .providers.base import ProviderResult

logger = logging.getLogger("llm")

# ─── LLM Result Types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMResult:
    """Rich return object from LLM calls, carrying metadata for telemetry."""
    text: str
    provider: str
    model: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: float
    fallback_depth: int


class StreamingLLMResult:
    """Async-iterable wrapper that accumulates metadata during streaming."""

    def __init__(
        self, stream, provider: str, model: str, fallback_depth: int
    ) -> None:
        self._stream = stream
        self._provider = provider
        self._model = model
        self._fallback_depth = fallback_depth
        self._text = ""
        self._start = _time_mod.monotonic()

    async def __aiter__(self):
        async for chunk in self._stream:
            self._text += chunk
            yield chunk
        from ..telemetry import get_current_tracker
        tracker = get_current_tracker()
        if tracker is not None:
            tracker.record_llm_result(self.metadata)

    @property
    def metadata(self) -> LLMResult:
        return LLMResult(
            text=self._text,
            provider=self._provider,
            model=self._model,
            tokens_in=None,
            tokens_out=None,
            latency_ms=(_time_mod.monotonic() - self._start) * 1000,
            fallback_depth=self._fallback_depth,
        )


# ─── Thin Wrappers ─────────────────────────────────────────────────────────


def _vision_yes_no_sync(image_base64: str, prompt: str) -> str:
    """
    Synchronous vision YES/NO check using Groq llama-4-scout.
    Used by the action executor (non-async context) to verify screen state.
    Returns "YES" or "NO". Returns "NO" on any error.
    """
    groq = provider_registry.get("groq")
    if groq and hasattr(groq, "vision_yes_no_sync"):
        return groq.vision_yes_no_sync(image_base64, prompt)
    return "NO"


# ─── Task Routing Tables ──────────────────────────────────────────────────

TASK_MODEL_MAP = {
    # Code generation: Flash's reasoning is strongest. Groq stays as defensive fallback.
    "code_gen": [
        ("gemini",   "gemini-2.5-flash"),
        ("groq",     "moonshotai/kimi-k2-instruct"),
        ("groq",     "llama-3.3-70b-versatile"),
        ("groq",     "qwen/qwen3-32b"),
    ],
    # Intent classification: Flash-Lite — fast, cheap, better than 8b-instant.
    "intent": [
        ("gemini",   "gemini-2.5-flash-lite"),
        ("groq",     "llama-3.1-8b-instant"),
    ],
    # Agent / planner calls: Flash for structured multi-step reasoning.
    "agent_plan": [
        ("gemini",   "gemini-2.5-flash"),
        ("groq",     "llama-3.3-70b-versatile"),
    ],
    # Nightly reflection: quality matters more than latency.
    "personality_reflection": [
        ("gemini",   "gemini-2.5-flash"),
        ("groq",     "llama-3.3-70b-versatile"),
    ],
    # Goal-achievement verification (text fallback path). Vision path uses Gemini natively.
    "agent_verify": [
        ("gemini",   "gemini-2.5-flash"),
        ("groq",     "meta-llama/llama-4-scout-17b-16e-instruct"),
    ],
    # Personality-bearing small talk: Flash to preserve personality nuance.
    "small_talk": [
        ("gemini",   "gemini-2.5-flash"),
        ("groq",     "llama-3.3-70b-versatile"),
    ],
    # Tool-result synthesis: Flash-Lite is plenty; Cerebras gpt-oss-120b as fallback.
    "synthesis": [
        ("gemini",   "gemini-2.5-flash-lite"),
        ("cerebras", "gpt-oss-120b"),
    ],
    # knowledge-graph entity/fact/relationship extraction — small JSON output, runs after
    # every turn that passes pre-filter. Flash-Lite is plenty; Cerebras gpt-oss-120b
    # keeps public forks usable without a Gemini key.
    "kg_extraction": [
        ("gemini",   "gemini-2.5-flash-lite"),
        ("cerebras", "gpt-oss-120b"),
        ("groq",     "llama-3.1-8b-instant"),
    ],
    # knowledge-graph D — multi-hop COT validator. Tiny JSON yes/no + optional name.
    # Cheap model is fine; loop caps iterations so cost is bounded.
    "kg_followup": [
        ("gemini",   "gemini-2.5-flash-lite"),
        ("cerebras", "gpt-oss-120b"),
        ("groq",     "llama-3.1-8b-instant"),
    ],
    # Default bucket — short utility calls.
    "default": [
        ("gemini",   "gemini-2.5-flash-lite"),
        ("cerebras", "gpt-oss-120b"),
    ],
}

# Deterministic tasks get temperature=0; creative tasks keep 0.7 default.
TASK_TEMPERATURE = {
    "intent": 0,
    "agent_plan": 0,
    "agent_verify": 0,
    "code_gen": 0,
    "kg_followup": 0,
}

# ─── Provider Chain (last-resort fallback after task-specific chain) ─────────

PROVIDERS = [
    # Gemini primary
    {"name": "gemini",   "model": "gemini-2.5-flash",         "api_key_env": "GEMINI_API_KEY"},
    {"name": "gemini",   "model": "gemini-2.5-flash-lite",    "api_key_env": "GEMINI_API_KEY"},
    # Groq free-tier fallback
    {"name": "groq",     "model": "moonshotai/kimi-k2-instruct", "api_key_env": "GROQ_API_KEY"},
    {"name": "groq",     "model": "llama-3.3-70b-versatile",     "api_key_env": "GROQ_API_KEY"},
    {"name": "groq",     "model": "llama-3.1-8b-instant",        "api_key_env": "GROQ_API_KEY"},
    # Cerebras free-tier fallback
    {"name": "cerebras", "model": "gpt-oss-120b",              "api_key_env": "CEREBRAS_API_KEY"},
    # Local fallback
    {"name": "ollama",   "model": "llama3.1:8b",              "api_key_env": None},
]


# ─── Public API ──────────────────────────────────────────────────────────────


async def get_llm_response(
    prompt: str,
    system_prompt: str | None = None,
    json_mode: bool = False,
    max_tokens: int = 256,
    task_type: str = "default",
    temperature: float | None = None,
    messages: list[dict] | None = None,
) -> LLMResult:
    """
    Send a prompt through the provider chain and return the first successful response.

    Returns an LLMResult with .text, .provider, .model, .latency_ms, .fallback_depth.
    Returns LLMResult with text="__LLM_UNAVAILABLE__" if all providers fail.
    """
    if system_prompt is None:
        from .prompts import build_personality_prompt
        system_prompt = build_personality_prompt()

    if json_mode:
        system_prompt = (
            system_prompt.rstrip()
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation."
        )

    if temperature is None:
        temperature = TASK_TEMPERATURE.get(task_type)

    _call_start = _time_mod.monotonic()
    _depth = 0

    # Route through task-specific model chain before generic fallback
    task_chain = TASK_MODEL_MAP.get(task_type, TASK_MODEL_MAP["default"])

    for preferred_provider, preferred_model in task_chain:
        provider = provider_registry.get(preferred_provider)
        if not provider:
            _depth += 1
            continue

        result = provider.chat(prompt, system_prompt, max_tokens,
                               model=preferred_model, temperature=temperature,
                               messages=messages)

        if result:
            llm_result = LLMResult(
                text=result.text, provider=preferred_provider, model=preferred_model,
                tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                latency_ms=(_time_mod.monotonic() - _call_start) * 1000,
                fallback_depth=_depth,
            )
            try:
                from ..telemetry import get_current_tracker
                tracker = get_current_tracker()
                if tracker:
                    tracker.record_llm_result(llm_result)
            except Exception:
                pass
            return llm_result

        _depth += 1
        logger.warning(f"[LLM] '{task_type}' model ({preferred_model}) on {preferred_provider} failed, trying next...")

    # Try each provider in order
    for prov_entry in PROVIDERS:
        name = prov_entry["name"]
        api_key_env = prov_entry["api_key_env"]

        # Skip if API key required but not set
        if api_key_env:
            api_key = getattr(config, api_key_env, "") or ""
            if not api_key:
                continue

        provider = provider_registry.get(name)
        if not provider:
            continue

        result = provider.chat(prompt, system_prompt, max_tokens,
                               model=prov_entry.get("model"), temperature=temperature,
                               messages=messages)

        if result:
            llm_result = LLMResult(
                text=result.text, provider=name, model=prov_entry.get("model", "unknown"),
                tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                latency_ms=(_time_mod.monotonic() - _call_start) * 1000,
                fallback_depth=_depth,
            )
            try:
                from ..telemetry import get_current_tracker
                tracker = get_current_tracker()
                if tracker:
                    tracker.record_llm_result(llm_result)
            except Exception:
                pass
            return llm_result

        _depth += 1
        logger.warning(f"[LLM] {name} failed, trying next provider...")

    # All providers failed
    logger.error("[LLM] All providers failed")
    return LLMResult(
        text="__LLM_UNAVAILABLE__", provider="none", model="none",
        tokens_in=None, tokens_out=None,
        latency_ms=(_time_mod.monotonic() - _call_start) * 1000,
        fallback_depth=_depth,
    )


async def _raw_stream(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    task_type: str = "default",
    temperature: float | None = None,
    messages: list[dict] | None = None,
):
    """Inner generator — yields raw token chunks."""
    if system_prompt is None:
        from .prompts import build_personality_prompt
        system_prompt = build_personality_prompt()

    if temperature is None:
        temperature = TASK_TEMPERATURE.get(task_type)

    task_chain = TASK_MODEL_MAP.get(task_type, TASK_MODEL_MAP["default"])

    for preferred_provider, preferred_model in task_chain:
        provider = provider_registry.get(preferred_provider)
        if not provider or not hasattr(provider, "stream"):
            continue

        kwargs = {
            "user_message": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "model": preferred_model,
            "temperature": temperature,
            "messages": messages,
        }

        yielded = False
        async for chunk in provider.stream(**kwargs):
            yielded = True
            yield chunk
        if yielded:
            return
        logger.warning(f"[LLM] Streaming '{task_type}' on {preferred_provider}/{preferred_model} failed, trying next...")

    logger.warning("[LLM] All streaming providers failed, falling back to non-streaming")
    result = await get_llm_response(prompt, system_prompt, task_type=task_type,
                                     max_tokens=max_tokens, temperature=temperature,
                                     messages=messages)
    if result.text and result.text != "__LLM_UNAVAILABLE__":
        yield result.text


async def get_llm_response_stream(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    task_type: str = "default",
    temperature: float | None = None,
    messages: list[dict] | None = None,
) -> StreamingLLMResult:
    """
    Streaming variant of get_llm_response(). Returns a StreamingLLMResult
    that can be async-iterated for token chunks. After iteration, call
    .metadata to get the LLMResult with accumulated text and timing.
    """
    task_chain = TASK_MODEL_MAP.get(task_type, TASK_MODEL_MAP["default"])
    provider, model = task_chain[0] if task_chain else ("gemini", "gemini-2.5-flash")
    return StreamingLLMResult(
        _raw_stream(prompt, system_prompt, max_tokens, task_type, temperature, messages),
        provider=provider, model=model, fallback_depth=0,
    )


async def chat(user_message: str, system_prompt: str | None = None, task_type: str = "default") -> str:
    """
    Send a message to the LLM and get a response.
    Backward-compatible wrapper around get_llm_response().

    Returns:
        The LLM's response text (or a fallback message on failure).
        Note: May contain an emotion tag prefix like "[happy] text" when
        called with task_type="small_talk". Caller should use
        parse_emotion_tag() to extract emotion and clean text.
    """
    result = await get_llm_response(user_message, system_prompt, task_type=task_type)
    if result.text == "__LLM_UNAVAILABLE__":
        return "[neutral] Hmph, my brain's not working right now. Try again in a sec."
    return result.text


async def get_vision_response(
    image_base64: str,
    prompt: str,
    system_prompt: str | None = None,
    json_mode: bool = False,
    max_tokens: int = 4096,
) -> LLMResult:
    """
    Send a screenshot image + text prompt to a vision-capable LLM.

    Primary:  Gemini Flash (unified multimodal)
    Fallback: Groq llama-4-scout-17b (vision capable, 30K TPM)
    Last resort: Text-only LLM with notice that image unavailable

    Args:
        image_base64: image bytes as base64
        prompt:       text question about the image
        system_prompt: optional system instruction
        json_mode:    when True, instruct the model to emit pure JSON
        max_tokens:   max OUTPUT tokens. Default 4096 covers planner JSON
                      with TODO progress + action history without truncation.
                      Bumped from 2048 on 2026-04-26 after a live test
                      truncation at 4929 total tokens (2.9k in + ~2k out)
                      hit the prior cap mid-string. Short-output callers
                      (yes/no visual-confirm) are unaffected — they emit
                      fewer tokens regardless of the ceiling.

    Returns LLMResult. text="__LLM_UNAVAILABLE__" on total failure.
    """
    _call_start = _time_mod.monotonic()

    if system_prompt is None:
        from .prompts import build_personality_prompt
        system_prompt = build_personality_prompt()

    if json_mode:
        system_prompt = (
            system_prompt.rstrip()
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation."
        )

    # Primary: Gemini (unified multimodal)
    gemini = provider_registry.get("gemini")
    if gemini and hasattr(gemini, "vision"):
        gemini_result = gemini.vision(
            image_base64, prompt, system_prompt=system_prompt, max_tokens=max_tokens
        )
        if gemini_result:
            llm_result = LLMResult(
                text=gemini_result.text, provider="gemini", model="gemini-2.5-flash",
                tokens_in=gemini_result.tokens_in, tokens_out=gemini_result.tokens_out,
                latency_ms=(_time_mod.monotonic() - _call_start) * 1000,
                fallback_depth=0,
            )
            try:
                from ..telemetry import get_current_tracker
                tracker = get_current_tracker()
                if tracker:
                    tracker.record_vision_call(llm_result)
            except Exception:
                pass
            return llm_result

    logger.warning("[llm] Gemini vision unavailable/failed — falling back to Groq llama-4-scout")

    # Fallback: Groq vision
    groq = provider_registry.get("groq")
    if groq and hasattr(groq, "vision"):
        try:
            groq_result = groq.vision(
                image_base64, prompt, system_prompt=system_prompt, max_tokens=max_tokens
            )
            if groq_result:
                llm_result = LLMResult(
                    text=groq_result.text, provider="groq", model="llama-4-scout-17b",
                    tokens_in=groq_result.tokens_in, tokens_out=groq_result.tokens_out,
                    latency_ms=(_time_mod.monotonic() - _call_start) * 1000,
                    fallback_depth=1,
                )
                try:
                    from ..telemetry import get_current_tracker
                    tracker = get_current_tracker()
                    if tracker:
                        tracker.record_vision_call(llm_result)
                except Exception:
                    pass
                return llm_result
        except Exception as e:
            logger.warning(f"[llm] WARNING: Vision failed, falling back to text-only LLM: {e}")

    fallback_prompt = "[Vision unavailable — reasoning from text only]\n" + prompt
    return await get_llm_response(fallback_prompt, system_prompt, json_mode)


# ─── Utility Functions ──────────────────────────────────────────────────────


def locate_element_bbox(text_description: str, screenshot_b64: str) -> tuple[int, int] | None:
    """
    Use Gemini's spatial-understanding (bbox detection) to locate a UI
    element by description. Returns (center_x, center_y) in SCREEN pixels,
    or None if Gemini cannot find the element.

    This is the precision fallback for vision_guided_click when OCR-snap
    fails — empty inputs without placeholder text, light-coloured buttons
    that fall below EasyOCR's confidence threshold, or LLM coords too far
    from the target for the snap radius. Bounding-box mode is purpose-built
    for this and is meaningfully more accurate than free-form pixel-coord
    estimation.

    Cost: one extra Gemini Flash vision call (~600-1500 tokens) per click
    that needed it. OCR-snap covers the common case for free.
    """
    import re as _re

    prompt = (
        f'Find the UI element on screen matching this description:\n'
        f'  "{text_description}"\n\n'
        f'Return ONLY a JSON object with the element\'s bounding box. The box is\n'
        f'normalized to integers 0-1000 in [ymin, xmin, ymax, xmax] order, where\n'
        f'image top-left is (0,0) and bottom-right is (1000,1000):\n'
        f'  {{"box_2d": [ymin, xmin, ymax, xmax]}}\n\n'
        f'If the element is NOT visible, return:\n'
        f'  {{"box_2d": null}}\n\n'
        f'Do not include any other text, explanation, or markdown.'
    )

    gemini = provider_registry.get("gemini")
    if not gemini or not hasattr(gemini, "vision"):
        return None

    _bbox_result = gemini.vision(
        image_base64=screenshot_b64,
        prompt=prompt,
        max_tokens=128,
    )
    if not _bbox_result:
        return None

    # Tolerate the LLM wrapping JSON in fences or extra prose.
    match = _re.search(r'\{[^{}]*"box_2d"[^{}]*\}', _bbox_result.text, _re.DOTALL)
    if not match:
        logger.debug(f"[llm] bbox: no JSON object in response: {_bbox_result.text[:120]!r}")
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.debug(f"[llm] bbox: JSON parse failed: {match.group(0)[:120]!r}")
        return None

    box = data.get("box_2d")
    if not box or not isinstance(box, list) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (int(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not (0 <= xmin < xmax <= 1000 and 0 <= ymin < ymax <= 1000):
        logger.debug(f"[llm] bbox: invalid range {box}")
        return None

    try:
        import pyautogui
        screen_w, screen_h = pyautogui.size()
    except Exception as e:
        logger.debug(f"[llm] pyautogui screen size unavailable: {e}")
        return None

    cx_norm = (xmin + xmax) / 2.0
    cy_norm = (ymin + ymax) / 2.0
    sx = int(cx_norm * screen_w / 1000)
    sy = int(cy_norm * screen_h / 1000)
    logger.info(
        f"[llm] bbox located '{text_description}' at ({sx},{sy}) "
        f"[norm box=({xmin},{ymin},{xmax},{ymax})]"
    )
    return (sx, sy)


async def classify_emotion(text: str) -> str:
    """
    Classify the emotional tone of a response text.
    Returns one of: neutral, happy, excited, sad, angry, sarcastic, worried, surprised.
    Uses llama-3.1-8b-instant — cheap and fast.
    Falls back to 'neutral' on any error.
    """
    if not text or not text.strip():
        return "neutral"
    try:
        result = await get_llm_response(
            text,
            system_prompt=(
                "Classify the emotional tone of this assistant response as exactly one word. "
                "Choose from: neutral, happy, excited, sad, angry, sarcastic, worried, surprised. "
                "Guidelines:\n"
                "- happy: cheerful, positive, warm responses\n"
                "- excited: enthusiastic, energetic, announcing something cool\n"
                "- sad: empathetic, comforting, bad news\n"
                "- angry: frustrated, firm, scolding\n"
                "- sarcastic: teasing, dry humor, playful jabs\n"
                "- worried: concerned, cautious, warning\n"
                "- surprised: shocked, amazed, unexpected\n"
                "- neutral: informational, factual, calm\n"
                "Reply with only that single word, lowercase, no punctuation."
            ),
            task_type="intent",   # reuses llama-3.1-8b-instant
            json_mode=False,
            max_tokens=5,
        )
        word = result.text.strip().lower().split()[0] if result.text else "neutral"
        # Map legacy "calm" -> "neutral" for backwards compatibility
        if word == "calm":
            word = "neutral"
        return word if word in config.VALID_EMOTIONS else "neutral"
    except Exception:
        return "neutral"


def parse_emotion_tag(text: str) -> tuple[str | None, str]:
    """
    Parse an inline emotion tag from the LLM response.

    Expected format: "[happy] response text here"
    Returns (emotion, clean_text) tuple.
    Returns (None, original_text) if no valid tag found — this lets callers
    distinguish "no tag" from "[neutral] text".
    """
    if not text or not text.strip():
        return None, text or ""

    stripped = text.strip()

    # Match [emotion] at the start
    if stripped.startswith("["):
        bracket_end = stripped.find("]")
        if bracket_end != -1:
            tag = stripped[1:bracket_end].strip().lower()
            if tag in config.VALID_EMOTIONS:
                clean = stripped[bracket_end + 1:].strip()
                return tag, clean

    return None, stripped


async def extract_facts(text: str) -> list[dict]:
    """
    Extract personal facts from a user message.
    Returns a list of {"key": str, "value": str} dicts.
    Returns empty list if no facts found or on any error.

    Only fires when the message contains personal information signals.
    Uses llama-3.1-8b-instant (same as intent/emotion — cheap and fast).
    """
    if not text or not text.strip():
        return []

    # Quick pre-filter — only call LLM if message likely contains personal info
    # This saves API calls on messages that clearly have no facts
    personal_signals = (
        "my ", "i am", "i'm", "i live", "i work", "i have", "i like",
        "i love", "i hate", "i study", "i go to", "call me", "name is",
        "age is", "i'm from", "i am from", "my name", "my age",
        "my phone", "my email", "my job", "my city", "my country"
    )
    lowered = text.lower()
    if not any(signal in lowered for signal in personal_signals):
        return []

    try:
        result = await get_llm_response(
            text,
            system_prompt=(
                "Extract personal facts about the user from this message. "
                "Return ONLY a JSON array of objects with 'key' and 'value' fields. "
                "Use snake_case keys like: user_name, user_age, user_location, "
                "user_job, user_hobby, user_phone, user_email, user_preference. "
                "If no personal facts are present, return an empty array: []. "
                "Examples:\n"
                "Input: 'My name is Alex and I live in Berlin'\n"
                "Output: [{\"key\": \"user_name\", \"value\": \"Alex\"}, "
                "{\"key\": \"user_location\", \"value\": \"Berlin\"}]\n"
                "Input: 'what is the weather today'\n"
                "Output: []\n"
                "IMPORTANT: Return ONLY the JSON array, nothing else."
            ),
            task_type="intent",
            json_mode=False,
            max_tokens=150,
        )

        # Parse the JSON array response
        import re
        result = result.text.strip()

        # Handle empty array fast
        if result == "[]":
            return []

        # Try direct parse
        if result.startswith("["):
            facts = json.loads(result)
        else:
            # Try to extract array from response
            match = re.search(r'\[.*?\]', result, re.DOTALL)
            if match:
                facts = json.loads(match.group(0))
            else:
                return []

        # Validate structure
        validated = []
        for f in facts:
            if isinstance(f, dict) and "key" in f and "value" in f:
                key = str(f["key"]).strip().lower().replace(" ", "_")
                value = str(f["value"]).strip()
                if key and value:
                    validated.append({"key": key, "value": value})

        return validated

    except Exception as e:
        logger.debug(f"[LLM] extract_facts failed (non-critical): {e}")
        return []
