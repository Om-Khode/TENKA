"""Tests for Context Compression + Native Multi-Turn Messages."""

from __future__ import annotations

import inspect


def test_provider_protocol_accepts_messages_param():
    """Provider.chat() signature includes messages param."""
    from assistant.llm.providers.base import Provider

    sig = inspect.signature(Provider.chat)
    assert "messages" in sig.parameters, "Provider.chat() must accept 'messages' param"
    param = sig.parameters["messages"]
    assert param.default is None, "messages param must default to None"


from unittest.mock import MagicMock, patch


def _make_history():
    """Helper: 3-turn conversation history."""
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you?"},
        {"role": "assistant", "content": "doing well!"},
    ]


# ─── Groq ───────────────────────────────────────────────────────────────────


class TestGroqMultiTurn:
    def test_chat_without_messages_uses_two_message_array(self):
        """Groq chat() with messages=None builds [system, user] as before."""
        from assistant.llm.providers.groq import GroqProvider

        provider = GroqProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("assistant.llm.providers.groq.Groq", return_value=mock_client), \
             patch("assistant.llm.providers.groq._current_groq_key", "fake-key"):
            result = provider.chat("hello", "be nice", max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_chat_with_messages_builds_full_array(self):
        """Groq chat() with messages builds [system] + history + [user]."""
        from assistant.llm.providers.groq import GroqProvider

        provider = GroqProvider()
        history = _make_history()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("assistant.llm.providers.groq.Groq", return_value=mock_client), \
             patch("assistant.llm.providers.groq._current_groq_key", "fake-key"):
            result = provider.chat("what's up?", "be nice", max_tokens=100, messages=history)

        call_kwargs = mock_client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(msgs) == 6  # system + 4 history + current user
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[1:5] == history
        assert msgs[5] == {"role": "user", "content": "what's up?"}


# ─── Cerebras ───────────────────────────────────────────────────────────────


class TestCerebrasMultiTurn:
    def test_chat_with_messages_builds_full_array(self):
        """Cerebras chat() with messages builds [system] + history + [user]."""
        from assistant.llm.providers.cerebras import CerebrasProvider

        provider = CerebrasProvider()
        history = _make_history()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("assistant.llm.providers.cerebras.Cerebras", return_value=mock_client), \
             patch("assistant.config.CEREBRAS_API_KEY", "fake-key"):
            result = provider.chat("what's up?", "be nice", max_tokens=100, messages=history)

        call_kwargs = mock_client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(msgs) == 6
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[1:5] == history
        assert msgs[5] == {"role": "user", "content": "what's up?"}


# ─── Ollama ─────────────────────────────────────────────────────────────────


class TestOllamaMultiTurn:
    def test_chat_with_messages_builds_full_array(self):
        """Ollama chat() with messages builds [system] + history + [user]."""
        from assistant.llm.providers.ollama import OllamaProvider

        provider = OllamaProvider()
        history = _make_history()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "test response"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch("assistant.llm.providers.ollama.requests.post", return_value=mock_resp) as mock_post:
            result = provider.chat("what's up?", "be nice", max_tokens=100, messages=history)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        msgs = payload["messages"]
        assert len(msgs) == 6
        assert msgs[0] == {"role": "system", "content": "be nice"}
        assert msgs[1:5] == history
        assert msgs[5] == {"role": "user", "content": "what's up?"}


# ─── Gemini ─────────────────────────────────────────────────────────────────


class TestGeminiMultiTurn:
    def test_chat_without_messages_passes_string_contents(self):
        """Gemini chat() without messages passes user_message as string."""
        from assistant.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider()

        mock_response = MagicMock()
        mock_response.text = "test response"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("assistant.llm.providers.gemini.genai", create=True) as mock_genai, \
             patch("assistant.llm.providers.gemini.types", create=True) as mock_types:
            mock_genai.Client.return_value = mock_client
            mock_types.HttpOptions = MagicMock()
            mock_types.GenerateContentConfig = MagicMock()
            mock_types.ThinkingConfig = MagicMock()

            with patch("assistant.config.GEMINI_API_KEY", "fake-key"):
                result = provider.chat("hello", "be nice", max_tokens=100)

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents")
        assert contents == "hello"

    def test_chat_with_messages_builds_contents_list(self):
        """Gemini chat() with messages builds contents[] with user/model roles."""
        from assistant.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        history = _make_history()

        mock_response = MagicMock()
        mock_response.text = "test response"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("assistant.llm.providers.gemini.genai", create=True) as mock_genai, \
             patch("assistant.llm.providers.gemini.types", create=True) as mock_types:
            mock_genai.Client.return_value = mock_client
            mock_types.HttpOptions = MagicMock()
            mock_types.GenerateContentConfig = MagicMock()
            mock_types.ThinkingConfig = MagicMock()
            # Make Content and Part trackable
            mock_types.Content = lambda **kw: kw
            mock_types.Part.from_text = lambda text: text

            with patch("assistant.config.GEMINI_API_KEY", "fake-key"):
                result = provider.chat("what's up?", "be nice", max_tokens=100, messages=history)

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents")
        assert isinstance(contents, list)
        assert len(contents) == 5  # 4 history entries + 1 current user


# ─── Router ─────────────────────────────────────────────────────────────────


class TestRouterPassthrough:
    def test_raw_stream_passes_messages_to_provider(self):
        """_raw_stream includes messages in provider kwargs."""
        import asyncio

        history = _make_history()

        async def run():
            # Mock a provider that captures its kwargs
            captured_kwargs = {}
            async def fake_stream(**kwargs):
                captured_kwargs.update(kwargs)
                yield "hello"

            mock_provider = MagicMock()
            mock_provider.stream = fake_stream

            with patch("assistant.llm.router.provider_registry") as mock_reg, \
                 patch("assistant.llm.router.TASK_MODEL_MAP", {"small_talk": [("gemini", "gemini-2.5-flash")]}):
                mock_reg.get.return_value = mock_provider

                from assistant.llm.router import _raw_stream
                chunks = []
                async for chunk in _raw_stream("test", "sys", task_type="small_talk", messages=history):
                    chunks.append(chunk)

            assert captured_kwargs.get("messages") is history

        asyncio.run(run())


# ─── Contracts ──────────────────────────────────────────────────────────────


class TestContracts:
    def test_stream_for_small_talk_accepts_messages(self):
        """stream_for_small_talk signature includes messages param."""
        import inspect
        from assistant.llm.contracts import stream_for_small_talk

        sig = inspect.signature(stream_for_small_talk)
        assert "messages" in sig.parameters

    def test_ask_for_context_compression_returns_summary(self):
        """ask_for_context_compression returns a summary string."""
        import asyncio

        turns = [
            {"user_input": "hello", "response": "hi there"},
            {"user_input": "what's AI?", "response": "artificial intelligence"},
        ]

        async def run():
            mock_result = MagicMock()
            mock_result.text = "User greeted the assistant and asked about AI."
            with patch("assistant.llm.contracts.get_llm_response", return_value=mock_result):
                from assistant.llm.contracts import ask_for_context_compression
                result = await ask_for_context_compression(turns)
                assert result == "User greeted the assistant and asked about AI."

        asyncio.run(run())

    def test_ask_for_context_compression_strips_code_fences(self):
        """Code fences in compression output are stripped."""
        import asyncio

        turns = [{"user_input": "test", "response": "test"}]

        async def run():
            mock_result = MagicMock()
            mock_result.text = "```\nSome summary text.\n```"
            with patch("assistant.llm.contracts.get_llm_response", return_value=mock_result):
                from assistant.llm.contracts import ask_for_context_compression
                result = await ask_for_context_compression(turns)
                assert result == "Some summary text."

        asyncio.run(run())


# ─── Core Message Builder ───────────────────────────────────────────────────


class TestBuildConversationMessages:
    @staticmethod
    def _make_turns(n, start_ts="2026-05-27T10:00:00"):
        """Generate n fake turns with incrementing timestamps."""
        from datetime import datetime, timedelta
        base = datetime.fromisoformat(start_ts)
        return [
            {
                "user_input": f"user msg {i}",
                "response": f"assistant msg {i}",
                "timestamp": (base + timedelta(minutes=i)).isoformat(),
            }
            for i in range(n)
        ]

    def test_empty_history_returns_empty(self):
        """No turns → empty messages, no summary."""
        import asyncio
        import assistant.main as main_mod

        async def run():
            with patch.object(main_mod, "memory") as mock_mem:
                mock_mem.get_recent.return_value = []
                messages, summary = await main_mod._build_conversation_messages()
                assert messages == []
                assert summary is None

        asyncio.run(run())

    def test_short_history_no_compression(self):
        """≤15 turns → all as messages, no compression."""
        import asyncio
        import assistant.main as main_mod

        turns = self._make_turns(10)

        async def run():
            with patch.object(main_mod, "memory") as mock_mem, \
                 patch.object(main_mod, "_get_personality_switch_ts", return_value=None), \
                 patch.object(main_mod, "_compression_cache", None):
                mock_mem.get_recent.return_value = turns
                messages, summary = await main_mod._build_conversation_messages()
                assert len(messages) == 20  # 10 turns × 2
                assert summary is None
                assert messages[0] == {"role": "user", "content": "user msg 0"}
                assert messages[1] == {"role": "assistant", "content": "assistant msg 0"}

        asyncio.run(run())

    def test_long_history_triggers_compression(self):
        """16+ turns → compress, keep last 10 verbatim."""
        import asyncio
        import assistant.main as main_mod

        turns = self._make_turns(20)

        async def run():
            with patch.object(main_mod, "memory") as mock_mem, \
                 patch.object(main_mod, "_get_personality_switch_ts", return_value=None), \
                 patch.object(main_mod, "_compression_cache", None), \
                 patch("assistant.llm.contracts.ask_for_context_compression",
                        return_value="Summary of early conversation.") as mock_compress:
                mock_mem.get_recent.return_value = turns
                messages, summary = await main_mod._build_conversation_messages()
                assert summary == "Summary of early conversation."
                assert len(messages) == 20  # last 10 turns × 2
                assert messages[0]["content"] == "user msg 10"
                mock_compress.assert_called_once()
                compressed_turns = mock_compress.call_args[0][0]
                assert len(compressed_turns) == 10

        asyncio.run(run())

    def test_personality_boundary_redacts_assistant(self):
        """Pre-switch assistant messages are redacted."""
        import asyncio
        import assistant.main as main_mod

        turns = self._make_turns(5, start_ts="2026-05-27T10:00:00")

        async def run():
            switch_ts = "2026-05-27T10:02:30"
            with patch.object(main_mod, "memory") as mock_mem, \
                 patch.object(main_mod, "_get_personality_switch_ts", return_value=switch_ts), \
                 patch.object(main_mod, "_compression_cache", None):
                mock_mem.get_recent.return_value = turns
                messages, summary = await main_mod._build_conversation_messages()

                # Turns 0,1,2 are before switch — redacted
                assert messages[1]["content"] == "(responded)"
                assert messages[3]["content"] == "(responded)"
                assert messages[5]["content"] == "(responded)"
                # Turns 3,4 are after switch — not redacted
                assert messages[7]["content"] == "assistant msg 3"
                assert messages[9]["content"] == "assistant msg 4"

        asyncio.run(run())

    def test_compression_cache_hit_skips_llm_call(self):
        """Cached summary is reused without calling LLM."""
        import asyncio
        import assistant.main as main_mod

        turns = self._make_turns(20)

        async def run():
            cache = {"summary": "Cached summary.", "compressed_up_to_index": 10}
            with patch.object(main_mod, "memory") as mock_mem, \
                 patch.object(main_mod, "_get_personality_switch_ts", return_value=None), \
                 patch.object(main_mod, "_compression_cache", cache), \
                 patch("assistant.llm.contracts.ask_for_context_compression") as mock_compress:
                mock_mem.get_recent.return_value = turns
                messages, summary = await main_mod._build_conversation_messages()
                assert summary == "Cached summary."
                mock_compress.assert_not_called()

        asyncio.run(run())


# ─── Slash Commands ─────────────────────────────────────────────────────────


class TestCompressCommand:
    def test_compress_is_recognized_as_slash_command(self):
        """'/compress' is recognized as a slash command."""
        from assistant.slash_commands import is_slash_command
        assert is_slash_command("/compress")

    def test_compress_is_in_reserved_set(self):
        """'compress' is in the RESERVED set."""
        from assistant.slash_commands import RESERVED
        assert "compress" in RESERVED

    def test_compress_clears_cache_and_returns_confirmation(self):
        """'/compress' clears compression cache and returns confirmation."""
        import assistant.main as main_mod
        from assistant.slash_commands import handle

        main_mod._compression_cache = {"summary": "old", "compressed_up_to_index": 5}
        result = handle("/compress")
        assert "compress" in result.lower() or "Fresh summary" in result
        assert main_mod._compression_cache is None


class TestCacheInvalidation:
    def test_personality_switch_clears_compression_cache(self):
        """Switching personality via /set clears the compression cache."""
        import assistant.main as main_mod

        main_mod._compression_cache = {"summary": "old", "compressed_up_to_index": 5}

        with patch("assistant.slash_commands.settings") as mock_settings, \
             patch("assistant.personality.switch_personality", return_value="Switched to warm_honest"):
            from assistant.slash_commands import _set_setting
            result = _set_setting("personality", "warm_honest")

        assert main_mod._compression_cache is None
        assert "Switched" in result


# ─── End-to-End ─────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_message_format_correctness(self):
        """Each turn produces exactly one user + one assistant message."""
        import asyncio
        import assistant.main as main_mod

        turns = [
            {"user_input": f"q{i}", "response": f"a{i}", "timestamp": f"2026-05-27T10:0{i}:00"}
            for i in range(5)
        ]

        async def run():
            with patch.object(main_mod, "memory") as mock_mem, \
                 patch.object(main_mod, "_get_personality_switch_ts", return_value=None), \
                 patch.object(main_mod, "_compression_cache", None):
                mock_mem.get_recent.return_value = turns
                messages, _ = await main_mod._build_conversation_messages()

                for i, msg in enumerate(messages):
                    expected_role = "user" if i % 2 == 0 else "assistant"
                    assert msg["role"] == expected_role, (
                        f"Message {i}: expected role '{expected_role}', got '{msg['role']}'"
                    )

        asyncio.run(run())

    def test_current_user_message_not_in_history(self):
        """The messages array contains only history, not the current user input."""
        import asyncio
        import assistant.main as main_mod

        turns = [
            {"user_input": "old question", "response": "old answer", "timestamp": "2026-05-27T10:00:00"},
        ]

        async def run():
            with patch.object(main_mod, "memory") as mock_mem, \
                 patch.object(main_mod, "_get_personality_switch_ts", return_value=None), \
                 patch.object(main_mod, "_compression_cache", None):
                mock_mem.get_recent.return_value = turns
                messages, _ = await main_mod._build_conversation_messages()
                assert len(messages) == 2
                contents = [m["content"] for m in messages]
                assert "old question" in contents
                assert "old answer" in contents

        asyncio.run(run())
