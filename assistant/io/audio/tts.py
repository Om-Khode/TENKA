"""
tts.py — Text-to-Speech module using the Kokoro Python library.

Generates speech audio locally (no Docker, no cloud) and plays it
through the system speakers via sounddevice.

Also notifies Unity (via the bridge) when talking starts/stops so
the avatar can animate lip-sync.

Kokoro is eagerly initialized at startup via init_tts() to avoid
first-use delay.

Vocal Voice — Single voicepack (af_heart) + CPU post-processing
(scipy resample pitch shift, EQ boost, tremolo, volume) for consistent
anime-style character voice across all emotions. Kokoro's native speed
param handles emotional pacing. No librosa dependency.
"""

import logging
import numpy as np
import sounddevice as sd
import warnings
import os

from ... import config

logger = logging.getLogger("tts")


# Track current voice to avoid unnecessary pipeline reloads
_current_voice: str = ""

# Eagerly-loaded Kokoro pipeline (set by init_tts)
_pipeline = None


def init_tts() -> bool:
    """
    Eagerly initialize the Kokoro TTS pipeline at startup.

    Suppresses noisy internal logs from Kokoro, phonemizer, and torch
    during loading, then logs a single success/failure line.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    global _pipeline

    if _pipeline is not None:
        return True  # Already initialized

    # Suppress warnings before kokoro import — they fire at import time
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress logging-based noise
    for noisy_logger in ("kokoro", "phonemizer", "torch", "espeak",
                        "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    try:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")  # 'a' = American English
        logger.info("[tts] Kokoro initialized successfully")

        # Log vocal voice status
        if config.VOCAL_VOICE_ENABLED:
            logger.info(f"[tts] Vocal voice ENABLED — base voice: {config.VOCAL_VOICE_BASE}")
        else:
            logger.info("[tts] Vocal voice DISABLED — using legacy voicepack swapping")

        return True

    except ImportError:
        logger.error(
            "[tts] Kokoro initialization failed: kokoro not installed "
            "— pip install kokoro"
        )
        return False
    except Exception as e:
        logger.error(f"[tts] Kokoro initialization failed: {e}")
        return False


def _format_for_tts(text: str) -> str:
    """
    Pre-process text before passing to Kokoro TTS.
    Converts long digit sequences (7+ digits) to space-separated digits
    so they are spoken naturally rather than as large numbers.
    e.g. "9876543210" → "9 8 7 6 5 4 3 2 1 0"
    """
    import re
    def _space_digits(match):
        return " ".join(list(match.group(0)))
    # Match standalone digit sequences of 7 or more digits
    return re.sub(r'\b\d{7,}\b', _space_digits, text)


def _preprocess_for_speech(text: str) -> str:
    """
    Clean up LLM-generated text for natural TTS output.
    
    Handles anime personality quirks that Kokoro reads badly:
    - Strips leftover emotion tags like [flustered], [annoyed] that aren't
      in the valid set (valid ones should already be stripped by parse_emotion_tag)
    - Converts stuttering: 'D-don't' → 'don't', 'I-it's' → 'it's'
      (TTS reads the hyphen as a pause and spells the stutter letter)
    - Replaces interjections with speakable equivalents:
      'Hmph' → 'Humph' (Kokoro spells H-M-P-H otherwise)
      'Tch' → removed (unspeakable consonant cluster)
    """
    import re
    
    # 1. Strip markdown formatting (LLM outputs *italic*, **bold**, etc.)
    text = re.sub(r'\*{2,}(.+?)\*{2,}', r'\1', text)  # **bold**
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', text)  # *italic*
    text = re.sub(r'~~(.+?)~~', r'\1', text)  # ~~strikethrough~~
    text = re.sub(r'\*', '', text)  # stray asterisks

    # 2. Remove any leftover emotion tags like [flustered] [annoyed]
    text = re.sub(r'^\[[\w]+\]\s*', '', text)
    
    # 3. Convert stuttering: drop the stutter letter before hyphen
    #    'D-don't' → 'don't', 'I-it's' → 'it's', 'B-baka' → 'baka'
    text = re.sub(r'(?<!\w)[A-Za-z]-([A-Za-z])', r'\1', text)

    # 4. Fix capitalization: after sentence-ending punctuation
    def _cap_after_punct(m):
        return m.group(1) + m.group(2).upper()
    text = re.sub(r'([.!?]\s+)([a-z])', _cap_after_punct, text)
    # Capitalize first character if lowercase
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # 5. Replace unspeakable interjections with TTS-friendly versions.
    # Anchor with word boundaries so we don't eat substrings inside real words
    # (e.g., 'tch ' inside "ca[tch t]hat" used to delete to "cathat").
    text = re.sub(r'\bHmph\b', 'Humph', text)
    text = re.sub(r'\bhmph\b', 'humph', text)
    # Tch! / tch, etc. — only when standalone interjection, then drop with
    # any immediately-following punctuation/whitespace so we don't leave
    # double spaces.
    text = re.sub(r'\b[Tt]ch\b[!.,]?\s*', '', text)
    # Collapse any whitespace that resulted from the deletion.
    text = re.sub(r'\s{2,}', ' ', text)
    
    # 6. Clean up whitespace
    text = re.sub(r'  +', ' ', text).strip()
    
    return text


def _pitch_shift_resample(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Pitch shift via scipy resample — fast, clean, no metallic artifacts.

    How it works: resampling to fewer samples and playing at the original
    sample rate raises the pitch. Duration gets shorter by factor
    1/2^(semitones/12), which is compensated by adjusting Kokoro's speed
    parameter in speak().

    Args:
        audio: numpy float32 array
        sr: sample rate (unused in computation but kept for API consistency)
        semitones: how many semitones to shift up (positive = higher pitch)

    Returns:
        Pitch-shifted numpy float32 array (shorter duration)
    """
    if semitones == 0:
        return audio

    from scipy.signal import resample

    factor = 2 ** (semitones / 12.0)
    new_len = int(len(audio) / factor)

    if new_len < 1:
        return audio

    return resample(audio, new_len).astype(np.float32)


def process_vocal_voice(audio: np.ndarray, sr: int, emotion: str = "neutral") -> np.ndarray:
    """
    Apply vocal voice character + emotion effects to Kokoro TTS output.

    Effects chain (all CPU-only, ~5-25ms total for 5s of audio):
      1. Pitch shift — scipy resample (fast, artifact-free)
      2. High EQ boost — brightness / anime quality
      3. Tremolo — voice shaking (sad/worried only)
      4. Volume — emotion intensity
      5. Clip — prevent distortion

    Note: Emotional pacing (speed) is NOT done here — it's handled by
    Kokoro's native speed parameter in speak() for natural-sounding results.

    Args:
        audio: numpy float32 array from Kokoro TTS
        sr: sample rate (24000 for Kokoro)
        emotion: emotion string from bridge expression

    Returns:
        Processed numpy float32 array, same sample rate
    """
    # Guard: empty or very short audio
    if audio is None or len(audio) < sr // 10:  # less than 100ms
        return audio if audio is not None else np.array([], dtype=np.float32)

    profile = config.EMOTION_VOICE_PROFILES.get(
        emotion, config.EMOTION_VOICE_PROFILES["neutral"]
    )

    # 1. Pitch shift (character identity + emotion)
    if profile["pitch"] != 0:
        audio = _pitch_shift_resample(audio, sr, profile["pitch"])

    # 2. High EQ boost (brightness / anime quality)
    if profile["eq_boost_db"] > 0:
        from scipy.signal import butter, sosfilt
        sos = butter(2, [2000, 6000], btype="band", fs=sr, output="sos")
        boosted = sosfilt(sos, audio)
        gain = 10 ** (profile["eq_boost_db"] / 20.0)
        audio = audio + boosted * (gain - 1.0)

    # 3. Tremolo (voice shaking for sad/worried)
    if profile["tremolo_hz"] > 0 and profile["tremolo_depth"] > 0:
        t = np.arange(len(audio)) / sr
        tremolo = 1.0 + profile["tremolo_depth"] * np.sin(
            2 * np.pi * profile["tremolo_hz"] * t
        )
        audio = audio * tremolo

    # 4. Volume (emotion intensity)
    audio = audio * profile["volume"]

    # 5. Clip to prevent distortion
    audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


async def speak(text: str, bridge=None, emotion: str = "neutral") -> bool:
    """
    Convert text to speech and play it through the speakers.

    Args:
        text:    The text to speak.
        bridge:  Optional UnityBridge instance — used to send
                 set_talking and show_subtitle commands to Unity.
        emotion: Emotion string from bridge expression value.

    Returns:
        True if audio was played successfully, False otherwise.
    """
    text = _format_for_tts(text)
    text = _preprocess_for_speech(text)

    if not text or not text.strip():
        logger.warning("Empty text, nothing to speak")
        return False

    logger.info(f'Speaking: "{text}"')

    # surface "Speaking" in the status pill — set RIGHT BEFORE the
    # audio actually plays (not at function entry) so the pill appears
    # coincident with the sound. Setting it before Kokoro synthesis would
    # leave the pill visible during the 1-2s of silent generation.
    _status_bus = None
    _StatusPhase = None
    try:
        from assistant.io.status_broadcaster import status as _status_bus, StatusPhase as _StatusPhase  # noqa: E501
    except Exception:
        _status_bus = None

    try:
        # Ensure pipeline is ready (fallback if init_tts wasn't called)
        if _pipeline is None:
            if not init_tts():
                logger.error("TTS not available — cannot speak")
                return False

        pipeline = _pipeline

        # ─── Voice + speed selection ──────────────────────────────────────
        global _current_voice

        if config.VOCAL_VOICE_ENABLED:
            # Single voicepack, all emotion expression via post-processing
            voice = config.VOCAL_VOICE_BASE

            profile = config.EMOTION_VOICE_PROFILES.get(
                emotion, config.EMOTION_VOICE_PROFILES["neutral"]
            )

            # Kokoro speed = emotion_speed / pitch_factor
            # - emotion_speed: emotional pacing (happy=1.15x faster, sad=0.85x slower)
            # - pitch_factor: compensates for duration shortening from resample pitch shift
            #   (shifting up by N semitones shortens audio by factor 2^(N/12))
            pitch_factor = 2 ** (profile["pitch"] / 12.0)
            speed = profile["speed"] / pitch_factor
        else:
            # Legacy: emotion-based voicepack swapping
            voice, speed = config.LEGACY_EMOTION_VOICE_MAP.get(
                emotion, (config.TTS_VOICE, config.TTS_SPEED)
            )

        if _current_voice != voice:
            logger.info(f"[tts] Voice: {voice}, vocal_mode: {config.VOCAL_VOICE_ENABLED}")
            _current_voice = voice

        # ─── Generate audio using Kokoro ──────────────────────────────────
        audio_chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            logger.error("Kokoro returned no audio")
            return False

        # Concatenate all audio chunks
        full_audio = np.concatenate(audio_chunks)

        # ─── Vocal voice post-processing ──────────────────────────────────
        if config.VOCAL_VOICE_ENABLED:
            full_audio = process_vocal_voice(
                full_audio, sr=config.TTS_SAMPLE_RATE, emotion=emotion
            )

        duration = len(full_audio) / config.TTS_SAMPLE_RATE
        logger.info(f"Generated {duration:.1f}s of audio")

        # Tell Unity we're about to start talking
        if bridge:
            await bridge.send_command("set_talking", value=True)
            await bridge.send_command("show_subtitle", text=text)

        # show Speaking pill NOW (synthesis is done, audio about to play)
        if _status_bus is not None:
            try: _status_bus.set(_StatusPhase.SPEAKING, detail=text[:48])
            except Exception: pass

        # Play audio on a thread so asyncio doesn't skip past sd.wait()
        import asyncio as _asyncio
        loop = _asyncio.get_running_loop()
        global _is_speaking
        _is_speaking = True
        try:
            await loop.run_in_executor(
                None,
                lambda: (sd.play(full_audio, samplerate=config.TTS_SAMPLE_RATE), sd.wait())
            )
        finally:
            _is_speaking = False

        # Tell Unity we stopped talking
        if bridge:
            await bridge.send_command("set_talking", value=False)

        # clear Speaking pill — overlay's Done flash takes care of the
        # brief "completed" UX before the pill hides.
        if _status_bus is not None:
            try: _status_bus.set(_StatusPhase.IDLE)
            except Exception: pass

        return True

    except Exception as e:
        logger.error(f"TTS error: {e}")
        _is_speaking = False
        # Make sure we reset the talking state even on error
        if bridge:
            try:
                await bridge.send_command("set_talking", value=False)
            except Exception:
                pass
        if _status_bus is not None:
            try: _status_bus.set(_StatusPhase.IDLE)
            except Exception: pass
        return False


_is_speaking = False


def is_speaking() -> bool:
    """Check if non-streaming audio is currently playing."""
    return _is_speaking
