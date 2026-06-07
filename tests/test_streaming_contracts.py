"""Tests for LLM streaming router and contracts."""

import asyncio
from types import SimpleNamespace
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for c in self._chunks:
            yield c


class TestStreamGemini:

    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        from assistant.llm.router import _stream_gemini

        mock_chunk_1 = MagicMock()
        mock_chunk_1.text = "Hello "
        mock_chunk_2 = MagicMock()
        mock_chunk_2.text = "world!"

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk_1, mock_chunk_2])

        with patch("assistant.llm.router.config") as mock_config:
            mock_config.GEMINI_API_KEY = "test-key"
            mock_config.LLM_TIMEOUT = 30
            with patch("google.genai.Client", return_value=mock_client):
                chunks = []
                async for chunk in _stream_gemini("test prompt", "system", 256,
                                                   model="gemini-2.5-flash"):
                    chunks.append(chunk)

        assert chunks == ["Hello ", "world!"]

    @pytest.mark.asyncio
    async def test_yields_nothing_on_no_api_key(self):
        from assistant.llm.router import _stream_gemini

        with patch("assistant.llm.router.config") as mock_config:
            mock_config.GEMINI_API_KEY = ""
            chunks = []
            async for chunk in _stream_gemini("test", "sys", 256):
                chunks.append(chunk)

        assert chunks == []


class TestStreamGroq:

    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        from assistant.llm.router import _stream_groq

        mock_delta_1 = MagicMock()
        mock_delta_1.choices = [MagicMock()]
        mock_delta_1.choices[0].delta.content = "Hello "
        mock_delta_2 = MagicMock()
        mock_delta_2.choices = [MagicMock()]
        mock_delta_2.choices[0].delta.content = "there!"
        mock_delta_3 = MagicMock()
        mock_delta_3.choices = [MagicMock()]
        mock_delta_3.choices[0].delta.content = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([
            mock_delta_1, mock_delta_2, mock_delta_3
        ])

        with patch("assistant.llm.router.config") as mock_config, \
             patch("assistant.llm.router._current_groq_key", "test-key"), \
             patch("groq.Groq", return_value=mock_client):
            mock_config.LLM_TIMEOUT = 30
            chunks = []
            async for chunk in _stream_groq("test", "sys", 256,
                                             model="llama-3.3-70b-versatile"):
                chunks.append(chunk)

        assert chunks == ["Hello ", "there!"]


class TestGetLlmResponseStream:

    @pytest.mark.asyncio
    async def test_falls_back_to_non_streaming(self):
        from assistant.llm.router import get_llm_response_stream

        async def empty_stream(*args, **kwargs):
            return
            yield

        with patch("assistant.llm.router._stream_gemini", side_effect=lambda *a, **kw: empty_stream()), \
             patch("assistant.llm.router._stream_groq", side_effect=lambda *a, **kw: empty_stream()), \
             patch("assistant.llm.router._stream_cerebras", side_effect=lambda *a, **kw: empty_stream()), \
             patch("assistant.llm.router._stream_ollama", side_effect=lambda *a, **kw: empty_stream()), \
             patch("assistant.llm.router.get_llm_response", new_callable=AsyncMock, return_value=SimpleNamespace(text="Fallback response")):

            chunks = []
            stream = await get_llm_response_stream("test", task_type="small_talk")
            async for chunk in stream:
                chunks.append(chunk)

        assert chunks == ["Fallback response"]

    @pytest.mark.asyncio
    async def test_uses_first_working_provider(self):
        from assistant.llm.router import get_llm_response_stream

        async def empty_stream(*args, **kwargs):
            return
            yield

        async def working_stream(*args, **kwargs):
            yield "Hello "
            yield "world!"

        with patch("assistant.llm.router._stream_gemini", side_effect=lambda *a, **kw: empty_stream()), \
             patch("assistant.llm.router._stream_groq", side_effect=lambda *a, **kw: working_stream()), \
             patch("assistant.llm.router.config") as mock_config:
            mock_config.LLM_SYSTEM_PROMPT = "test"
            chunks = []
            stream = await get_llm_response_stream("test", task_type="small_talk")
            async for chunk in stream:
                chunks.append(chunk)

        assert chunks == ["Hello ", "world!"]


class TestStreamingContracts:

    @pytest.mark.asyncio
    async def test_stream_for_small_talk_delegates(self):
        from assistant.llm.contracts import stream_for_small_talk

        with patch("assistant.llm.contracts.get_llm_response_stream",
                   new_callable=AsyncMock, return_value=_FakeStream(["chunk1", "chunk2"])):
            chunks = []
            async for chunk in stream_for_small_talk("hello"):
                chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_stream_for_synthesis_delegates(self):
        from assistant.llm.contracts import stream_for_synthesis

        with patch("assistant.llm.contracts.get_llm_response_stream",
                   new_callable=AsyncMock, return_value=_FakeStream(["result"])):
            chunks = []
            async for chunk in stream_for_synthesis("summarize this"):
                chunks.append(chunk)

        assert chunks == ["result"]
