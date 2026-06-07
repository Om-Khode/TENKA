"""Tests for the three-stage streaming TTS pipeline."""

import asyncio
import time as _time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call


async def _token_stream(tokens: list[str]):
    for t in tokens:
        yield t


class TestAccumulateSentences:

    @pytest.mark.asyncio
    async def test_sentences_queued_in_order(self):
        from assistant.io.audio.streaming import _accumulate_sentences
        from assistant.io.audio.sentence_buffer import SentenceBuffer

        queue = asyncio.Queue()
        buf = SentenceBuffer(min_length=0)
        tokens = ["First sentence. ", "Second sentence. ", "Third."]

        await _accumulate_sentences(_token_stream(tokens), queue, buf)

        sentences = []
        while not queue.empty():
            item = await queue.get()
            if item is None:
                break
            sentences.append(item)

        assert "First sentence." in sentences
        assert "Second sentence." in sentences
        assert any("Third" in s for s in sentences)

    @pytest.mark.asyncio
    async def test_sentinel_at_end(self):
        from assistant.io.audio.streaming import _accumulate_sentences
        from assistant.io.audio.sentence_buffer import SentenceBuffer

        queue = asyncio.Queue()
        buf = SentenceBuffer()

        await _accumulate_sentences(_token_stream(["Hello."]), queue, buf)

        items = []
        while not queue.empty():
            items.append(await queue.get())
        assert items[-1] is None


class TestTtsWorker:

    @pytest.mark.asyncio
    async def test_produces_audio_tuples(self):
        from assistant.io.audio.streaming import _tts_worker

        sentence_queue = asyncio.Queue()
        audio_queue = asyncio.Queue()
        await sentence_queue.put("Hello there.")
        await sentence_queue.put(None)

        fake_audio = np.zeros(24000, dtype=np.float32)

        with patch("assistant.io.audio.streaming._pipeline") as mock_pipeline, \
             patch("assistant.io.audio.streaming.process_vocal_voice", return_value=fake_audio), \
             patch("assistant.io.audio.streaming.config") as mock_config:
            mock_config.VOCAL_VOICE_ENABLED = True
            mock_config.VOCAL_VOICE_BASE = "af_heart"
            mock_config.EMOTION_VOICE_PROFILES = {
                "neutral": {"pitch": 0, "speed": 1.0, "volume": 1.0,
                            "tremolo_hz": 0, "tremolo_depth": 0, "eq_boost_db": 0}
            }
            mock_config.TTS_SAMPLE_RATE = 24000
            mock_pipeline.return_value = [(None, None, fake_audio)]

            await _tts_worker(sentence_queue, audio_queue,
                              voice="af_heart", speed=1.0, emotion="neutral")

        item = await audio_queue.get()
        assert isinstance(item, tuple)
        assert item[0] == "Hello there."
        assert isinstance(item[1], np.ndarray)

        sentinel = await audio_queue.get()
        assert sentinel is None


class TestAudioPlayer:

    @pytest.mark.asyncio
    async def test_plays_audio_and_updates_subtitle(self):
        from assistant.io.audio.streaming import _audio_player

        audio_queue = asyncio.Queue()
        fake_audio = np.zeros(24000, dtype=np.float32)
        await audio_queue.put(("Hello.", fake_audio))
        await audio_queue.put(None)

        mock_bridge = AsyncMock()

        with patch("assistant.io.audio.streaming._play_interruptible") as mock_play, \
             patch("assistant.io.audio.streaming.config") as mock_config:
            mock_config.TTS_SAMPLE_RATE = 24000

            await _audio_player(audio_queue, mock_bridge)

        mock_bridge.send_command.assert_any_call("set_talking", value=True)
        mock_bridge.send_command.assert_any_call("show_subtitle", text="Hello.")
        mock_bridge.send_command.assert_any_call("set_talking", value=False)
        mock_play.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_bridge_still_plays(self):
        from assistant.io.audio.streaming import _audio_player

        audio_queue = asyncio.Queue()
        fake_audio = np.zeros(24000, dtype=np.float32)
        await audio_queue.put(("Hello.", fake_audio))
        await audio_queue.put(None)

        with patch("assistant.io.audio.streaming._play_interruptible") as mock_play, \
             patch("assistant.io.audio.streaming.config") as mock_config:
            mock_config.TTS_SAMPLE_RATE = 24000

            await _audio_player(audio_queue, bridge=None)

        mock_play.assert_called_once()


class TestSpeakStreaming:

    @pytest.mark.asyncio
    async def test_returns_success_and_full_text(self):
        from assistant.io.audio import streaming

        fake_audio = np.zeros(24000, dtype=np.float32)

        with patch.object(streaming, "_pipeline") as mock_pipeline, \
             patch.object(streaming, "process_vocal_voice", return_value=fake_audio), \
             patch.object(streaming, "_play_interruptible") as mock_play, \
             patch.object(streaming, "config") as mock_config:
            mock_config.VOCAL_VOICE_ENABLED = True
            mock_config.VOCAL_VOICE_BASE = "af_heart"
            mock_config.EMOTION_VOICE_PROFILES = {
                "neutral": {"pitch": 0, "speed": 1.0, "volume": 1.0,
                            "tremolo_hz": 0, "tremolo_depth": 0, "eq_boost_db": 0}
            }
            mock_config.TTS_SAMPLE_RATE = 24000
            mock_pipeline.return_value = [(None, None, fake_audio)]

            tokens = ["Hello there. ", "How are you?"]
            success, text = await streaming.speak_streaming(
                _token_stream(tokens), bridge=None, emotion="neutral"
            )

        assert success is True
        assert "Hello" in text
        assert "How are you" in text

    @pytest.mark.asyncio
    async def test_empty_stream_returns_false(self):
        from assistant.io.audio import streaming

        async def empty():
            return
            yield

        success, text = await streaming.speak_streaming(
            empty(), bridge=None, emotion="neutral"
        )
        assert success is False
        assert text == ""


class TestPlayInterruptible:

    def test_plays_full_audio_when_not_interrupted(self):
        from assistant.io.audio import streaming

        streaming._stop_event.clear()
        fake_audio = np.zeros(2400, dtype=np.float32)  # 0.1s at 24kHz

        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.close = MagicMock()

        # Simulate finished_callback being called when stream starts
        def fake_init(**kwargs):
            # Store finished_callback, call it on start()
            cb = kwargs.get("finished_callback")
            inst = MagicMock()
            inst.start = lambda: cb() if cb else None
            inst.close = MagicMock()
            inst.abort = MagicMock()
            return inst

        with patch("assistant.io.audio.streaming.sd.OutputStream", side_effect=fake_init):
            streaming._play_interruptible(fake_audio, 24000)
        # Should complete without hanging

    def test_aborts_when_stop_event_set(self):
        from assistant.io.audio import streaming
        import threading

        streaming._stop_event.clear()
        fake_audio = np.zeros(240000, dtype=np.float32)  # 10s at 24kHz

        abort_called = threading.Event()

        def fake_init(**kwargs):
            inst = MagicMock()
            inst.start = MagicMock()
            inst.close = MagicMock()
            def _abort():
                abort_called.set()
            inst.abort = _abort
            # Never call finished_callback — simulates ongoing playback
            return inst

        with patch("assistant.io.audio.streaming.sd.OutputStream", side_effect=fake_init):
            # Set stop_event after a short delay
            def _set_stop():
                _time.sleep(0.05)
                streaming._stop_event.set()
            t = threading.Thread(target=_set_stop)
            t.start()

            streaming._play_interruptible(fake_audio, 24000)
            t.join()

        assert abort_called.is_set(), "stream.abort() should have been called"
        streaming._stop_event.clear()


class TestBargeIn:

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        from assistant.io.audio import streaming

        async def slow_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(slow_task())
        streaming._active_tasks = [task]
        streaming._stop_event.clear()

        await streaming.stop_streaming()

        # stop_streaming() schedules the cancel; let one event-loop tick run so
        # the CancelledError propagates and the task transitions from
        # "cancelling" to "cancelled".
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert task.cancelled()
        assert streaming._stop_event.is_set()
        assert streaming._active_tasks == []

    def test_is_speaking(self):
        from assistant.io.audio import streaming
        streaming._active_tasks = []
        assert streaming.is_speaking() is False
        streaming._active_tasks = [MagicMock()]
        assert streaming.is_speaking() is True
        streaming._active_tasks = []
