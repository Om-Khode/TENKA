"""Streaming TTS pipeline — three-stage asyncio.Queue architecture.

Stage 1 (Accumulator): LLM token stream → SentenceBuffer → sentence queue
Stage 2 (TTS Worker):  sentence queue → Kokoro generation → audio queue
Stage 3 (Audio Player): audio queue → sounddevice playback + bridge subtitles

Barge-in: stop_streaming() sets a threading.Event that the audio player
polls — sd.stop() is called from the SAME thread as sd.play(), avoiding
concurrent PortAudio access.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator

import numpy as np
import sounddevice as sd

from ... import config
from .sentence_buffer import SentenceBuffer
from .tts import (
    _format_for_tts,
    _preprocess_for_speech,
    _pipeline,
    init_tts,
    process_vocal_voice,
)

logger = logging.getLogger("streaming")

_active_tasks: list[asyncio.Task] = []
_stop_event = threading.Event()


# ─── Stage 1: Sentence Accumulator ───────────────────────────────────────────


async def _accumulate_sentences(
    token_stream: AsyncGenerator[str, None],
    sentence_queue: asyncio.Queue,
    buffer: SentenceBuffer,
):
    try:
        async for token in token_stream:
            for sentence in buffer.add(token):
                await sentence_queue.put(sentence)
                logger.info(f'[streaming] Sentence ready: "{sentence[:50]}..." ({len(sentence)} chars)')

        remainder = buffer.flush()
        if remainder:
            await sentence_queue.put(remainder)
            logger.info(f'[streaming] Final sentence: "{remainder[:50]}..." ({len(remainder)} chars)')
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[streaming] Accumulator error: {e}")
        remainder = buffer.flush()
        if remainder:
            try:
                await sentence_queue.put(remainder)
            except BaseException:
                pass
    finally:
        try:
            await sentence_queue.put(None)
        except BaseException:
            pass


# ─── Stage 2: TTS Worker ─────────────────────────────────────────────────────


async def _tts_worker(
    sentence_queue: asyncio.Queue,
    audio_queue: asyncio.Queue,
    voice: str,
    speed: float,
    emotion: str,
):
    try:
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break

            text = _preprocess_for_speech(_format_for_tts(sentence))
            if not text or not text.strip():
                continue

            pipeline = _pipeline
            if pipeline is None:
                if not init_tts():
                    logger.error("[streaming] TTS not available")
                    continue
                from .tts import _pipeline as _refreshed
                pipeline = _refreshed

            def _generate(p=pipeline, t=text, v=voice, s=speed):
                chunks = []
                for _, _, audio in p(t, voice=v, speed=s):
                    if audio is not None:
                        chunks.append(audio)
                return np.concatenate(chunks) if chunks else None

            audio = await asyncio.to_thread(_generate)
            if audio is None:
                logger.error(f"[streaming] Kokoro returned no audio for: {text[:50]}")
                continue

            if config.VOCAL_VOICE_ENABLED:
                audio = process_vocal_voice(audio, sr=config.TTS_SAMPLE_RATE, emotion=emotion)

            duration = len(audio) / config.TTS_SAMPLE_RATE
            logger.info(f'[streaming] Audio generated: {duration:.1f}s for "{text[:50]}..."')

            await audio_queue.put((sentence, audio))

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[streaming] TTS worker error: {e}")
    finally:
        try:
            await audio_queue.put(None)
        except BaseException:
            pass


# ─── Audio Playback (interruptible) ─────────────────────────────────────────


def _play_interruptible(audio: np.ndarray, sr: int) -> None:
    """Play audio via an explicit OutputStream, polling _stop_event for abort.

    We create the OutputStream ourselves so we hold a direct reference to
    it — sd.play() does not expose the stream it creates, so the old
    getattr(sd, "_last_stream") approach returned None every time.
    """
    data = audio.reshape(-1, 1) if audio.ndim == 1 else audio
    duration = len(data) / sr
    finished = threading.Event()
    position = [0]

    def _callback(outdata, frames, _time_info, _status):
        end = position[0] + frames
        if end >= len(data):
            chunk = len(data) - position[0]
            outdata[:chunk] = data[position[0]:]
            outdata[chunk:] = 0
            raise sd.CallbackStop()
        outdata[:] = data[position[0]:end]
        position[0] = end

    stream = sd.OutputStream(
        samplerate=sr,
        channels=data.shape[1],
        dtype=data.dtype,
        callback=_callback,
        finished_callback=finished.set,
    )
    stream.start()

    while not finished.is_set():
        if _stop_event.is_set():
            stream.abort()
            stream.close()
            return
        finished.wait(0.01)

    stream.close()


# ─── Stage 3: Audio Player ───────────────────────────────────────────────────


async def _audio_player(
    audio_queue: asyncio.Queue,
    bridge,
):
    first_chunk = True
    try:
        while True:
            item = await audio_queue.get()
            if item is None:
                break

            text, audio = item

            if _stop_event.is_set():
                continue

            if first_chunk and bridge:
                await bridge.send_command("set_talking", value=True)
                first_chunk = False

            if bridge:
                await bridge.send_command("show_subtitle", text=text)

            # refresh the Speaking pill detail with the current sentence.
            try:
                from assistant.io.status_broadcaster import status as _status_bus, StatusPhase as _StatusPhase
                _status_bus.set(_StatusPhase.SPEAKING, detail=text[:48])
            except Exception:
                pass

            logger.info(f'[streaming] Playing: "{text[:50]}..."')

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                _play_interruptible, audio, config.TTS_SAMPLE_RATE,
            )

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[streaming] Player error: {e}")
    finally:
        if bridge:
            try:
                await bridge.send_command("set_talking", value=False)
            except BaseException:
                pass


# ─── Orchestrator ─────────────────────────────────────────────────────────────


async def speak_streaming(
    token_stream: AsyncGenerator[str, None],
    bridge=None,
    emotion: str = "neutral",
) -> tuple[bool, str]:
    global _active_tasks

    _stop_event.clear()

    # do NOT pre-emit SPEAKING here — Kokoro takes ~1-2s to synthesize
    # the first chunk, and showing the Speaking pill during that silent gap
    # looks like a delay. The audio player loop below sets SPEAKING right
    # when each sentence is about to play (see _audio_player).
    _status_bus = None
    _StatusPhase = None
    try:
        from assistant.io.status_broadcaster import status as _status_bus, StatusPhase as _StatusPhase  # noqa: E501
    except Exception:
        _status_bus = None

    if config.VOCAL_VOICE_ENABLED:
        voice = config.VOCAL_VOICE_BASE
        profile = config.EMOTION_VOICE_PROFILES.get(
            emotion, config.EMOTION_VOICE_PROFILES["neutral"]
        )
        pitch_factor = 2 ** (profile["pitch"] / 12.0)
        speed = profile["speed"] / pitch_factor
    else:
        voice, speed = config.LEGACY_EMOTION_VOICE_MAP.get(
            emotion, (config.TTS_VOICE, config.TTS_SPEED)
        )

    sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    buffer = SentenceBuffer()
    collected_text: list[str] = []

    async def _collecting_stream():
        async for token in token_stream:
            collected_text.append(token)
            yield token

    task1 = asyncio.create_task(
        _accumulate_sentences(_collecting_stream(), sentence_queue, buffer)
    )
    task2 = asyncio.create_task(
        _tts_worker(sentence_queue, audio_queue, voice, speed, emotion)
    )
    task3 = asyncio.create_task(
        _audio_player(audio_queue, bridge)
    )

    _active_tasks = [task1, task2, task3]

    try:
        await asyncio.gather(task1, task2, task3)
    except BaseException as e:
        if not isinstance(e, asyncio.CancelledError):
            logger.error(f"[streaming] Pipeline error: {e}")
        for t in _active_tasks:
            if not t.done():
                t.cancel()
        _active_tasks = []
        full = "".join(collected_text)
        if _status_bus is not None:
            try: _status_bus.set(_StatusPhase.IDLE)
            except Exception: pass
        return False, full

    _active_tasks = []
    full = "".join(collected_text)
    if _status_bus is not None:
        try: _status_bus.set(_StatusPhase.IDLE)
        except Exception: pass

    if not full.strip():
        return False, ""

    return True, full


# ─── Barge-in ─────────────────────────────────────────────────────────────────


async def stop_streaming():
    """Stop both streaming AND one-shot TTS playback.

    Streaming: sets the shared _stop_event (read by _play_interruptible
    and _audio_player) and cancels the active pipeline tasks.

    One-shot (tts.speak): the audio is queued via sounddevice.play(); we
    call sounddevice.stop() to abort it immediately. This is what makes
    ESC silence small-talk replies, proactive nudges, and reminders
    (none of which go through the streaming pipeline).
    """
    global _active_tasks
    _stop_event.set()
    for task in _active_tasks:
        if not task.done():
            task.cancel()
    _active_tasks = []
    # Kill any one-shot sounddevice playback (non-streaming TTS path).
    try:
        sd.stop()
    except Exception:
        pass


def is_speaking() -> bool:
    return len(_active_tasks) > 0
