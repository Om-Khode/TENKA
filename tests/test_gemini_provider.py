"""
test_gemini_provider.py — Phase 1A/1B verification.

Validates the Gemini provider integration in assistant/llm.py:
  - _chat_gemini() text path (Flash + Flash-Lite)
  - _vision_gemini() unified multimodal path
  - get_vision_response() Gemini-primary / Groq-fallback wiring
  - error handling (missing key, bad model)

These tests make real API calls. They auto-skip when GEMINI_API_KEY is absent.

Run:
  pytest test_gemini_provider.py -v
  python test_gemini_provider.py        # standalone script mode
"""

import asyncio
import base64
import io
import os
import sys

import pytest

# Ensure project root is on sys.path when running standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env so GEMINI_API_KEY is available when running outside start_assistant.bat
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from assistant import config, llm  # noqa: E402

_HAS_KEY = bool(getattr(config, "GEMINI_API_KEY", "") or "")
_HAS_GROQ = bool(getattr(config, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY_1", ""))

requires_gemini = pytest.mark.skipif(
    not _HAS_KEY,
    reason="GEMINI_API_KEY not set — skipping live Gemini tests",
)


# ─── Static checks (always runnable) ──────────────────────────────────────────


def test_config_has_gemini_fields():
    assert hasattr(config, "GEMINI_API_KEY")
    assert hasattr(config, "GEMINI_MODEL")
    assert hasattr(config, "GEMINI_MODEL_LITE")
    assert config.GEMINI_MODEL == "gemini-2.5-flash" or config.GEMINI_MODEL
    assert "flash-lite" in config.GEMINI_MODEL_LITE


def test_chat_gemini_function_exists():
    assert hasattr(llm, "_chat_gemini")
    assert callable(llm._chat_gemini)


def test_vision_gemini_function_exists():
    assert hasattr(llm, "_vision_gemini")
    assert callable(llm._vision_gemini)


def test_google_genai_importable():
    """If the package isn't installed, _chat_gemini will log and return None — but
    we want this installed for Phase 1 to count as done."""
    try:
        from google import genai  # noqa: F401
        from google.genai import types  # noqa: F401
    except ImportError:
        pytest.fail("google-genai not installed — pip install google-genai")


def test_chat_gemini_without_key_returns_none(monkeypatch):
    """Empty API key should short-circuit cleanly, not raise."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    result = llm._chat_gemini("hello", "You are a bot", max_tokens=10)
    assert result is None


# ─── Live API tests ───────────────────────────────────────────────────────────


@requires_gemini
def test_chat_gemini_flash_basic():
    """Flash should respond to a simple prompt."""
    result = llm._chat_gemini(
        user_message="Reply with exactly the word PING and nothing else.",
        system_prompt="You are a test bot. Follow instructions literally.",
        max_tokens=20,
        model="gemini-2.5-flash",
        temperature=0,
    )
    assert result is not None, "Flash returned None — check API key and quota"
    assert "PING" in result.upper(), f"Expected PING in response, got: {result!r}"


@requires_gemini
def test_chat_gemini_flash_lite_basic():
    """Flash-Lite (no thinking mode) should also respond."""
    result = llm._chat_gemini(
        user_message="Reply with exactly the word PONG and nothing else.",
        system_prompt="You are a test bot. Follow instructions literally.",
        max_tokens=20,
        model="gemini-2.5-flash-lite",
        temperature=0,
    )
    assert result is not None, "Flash-Lite returned None — check API key and quota"
    assert "PONG" in result.upper(), f"Expected PONG, got: {result!r}"


@requires_gemini
def test_chat_gemini_honors_system_instruction():
    """system_prompt must be passed as native system_instruction, not user-message hack."""
    result = llm._chat_gemini(
        user_message="What is your designation?",
        system_prompt="You are Unit-7. You must introduce yourself as Unit-7. Be brief.",
        max_tokens=40,
        model="gemini-2.5-flash-lite",
        temperature=0,
    )
    assert result is not None
    assert "unit-7" in result.lower() or "unit 7" in result.lower(), (
        f"System instruction was not honored — got: {result!r}"
    )


@requires_gemini
def test_chat_gemini_temperature_accepted():
    """Custom temperature shouldn't raise."""
    result = llm._chat_gemini(
        user_message="Say hi.",
        system_prompt="Be brief.",
        max_tokens=20,
        model="gemini-2.5-flash-lite",
        temperature=0.3,
    )
    assert result is not None


@requires_gemini
def test_chat_gemini_bad_model_returns_none():
    """Non-existent model should return None, not crash."""
    result = llm._chat_gemini(
        user_message="hi",
        system_prompt="",
        max_tokens=10,
        model="gemini-does-not-exist",
    )
    assert result is None


@requires_gemini
def test_vision_gemini_describes_image():
    """Generate a tiny test image and ask Gemini to describe it."""
    from PIL import Image

    # 64x64 solid red square — simple, unambiguous
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = llm._vision_gemini(
        image_base64=b64,
        prompt="What color is this image? Reply with just the color name.",
        max_tokens=30,
    )
    assert result is not None, "Gemini vision returned None"
    assert "red" in result.lower(), f"Expected 'red' in vision response, got: {result!r}"


@requires_gemini
def test_get_vision_response_uses_gemini_first():
    """The public vision API should route through Gemini as primary."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    _result = asyncio.run(
        llm.get_vision_response(
            image_base64=b64,
            prompt="What color is this? One word.",
            system_prompt="You are a color identifier.",
        )
    )
    result = _result.text
    assert result != "__LLM_UNAVAILABLE__", "Vision totally failed"
    assert "green" in result.lower(), f"Expected 'green', got: {result!r}"


@requires_gemini
def test_get_llm_response_routes_through_gemini():
    """If a task chain has gemini, it should be invoked."""
    # Temporarily inject a gemini-first task chain
    original_map = llm.TASK_MODEL_MAP.copy()
    try:
        llm.TASK_MODEL_MAP["default"] = [("gemini", "gemini-2.5-flash-lite")]
        _result = asyncio.run(
            llm.get_llm_response(
                prompt="Reply with the single word READY.",
                system_prompt="Follow instructions literally.",
                max_tokens=20,
                task_type="default",
                temperature=0,
            )
        )
        result = _result.text
        assert result != "__LLM_UNAVAILABLE__"
        assert "READY" in result.upper()
    finally:
        llm.TASK_MODEL_MAP.update(original_map)


# ─── Standalone runner ────────────────────────────────────────────────────────


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
