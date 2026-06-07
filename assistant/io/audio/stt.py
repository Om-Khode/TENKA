"""
stt.py — Speech-to-Text module for the TENKA Voice Assistant.

Supports two backends (configurable in config.py):
  1. "whisper_cpp"    — Calls your existing whisper.cpp HTTP server.
                        Reuses the ggml-base.bin model you already have.
  2. "faster_whisper" — Uses the faster-whisper Python library (CTranslate2).
                        Downloads its own model on first run.

Both backends:
  - Record audio from the microphone using sounddevice
  - Save to a temporary WAV file
  - Transcribe and return the text
"""

import io
import time as _time
import wave
import logging
import tempfile
import requests
import numpy as np
import sounddevice as sd

from ... import config

logger = logging.getLogger("stt")

# Lazy-loaded faster-whisper model (only if that backend is selected)
_fw_model = None


def _get_faster_whisper_model():
    """Load the faster-whisper model (lazy, first call only)."""
    global _fw_model
    if _fw_model is None:
        logger.info(f"Loading faster-whisper model '{config.FASTER_WHISPER_MODEL}' ...")
        from faster_whisper import WhisperModel
        _fw_model = WhisperModel(
            config.FASTER_WHISPER_MODEL,
            device=config.FASTER_WHISPER_DEVICE,
            compute_type=config.FASTER_WHISPER_COMPUTE_TYPE,
        )
        logger.info("faster-whisper model loaded")
    return _fw_model


class Recorder:
    """
    Simple microphone recorder using sounddevice.
    Records 16 kHz mono audio into a numpy array.
    """

    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self.is_recording = False

    def start(self):
        """Start recording from the default microphone."""
        if self.is_recording:
            logger.warning("Already recording")
            return

        self._frames = []

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Sounddevice status: {status}")
            # indata is a 2D numpy array (frames x channels), copy it
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=config.RECORD_SAMPLE_RATE,
            channels=config.RECORD_CHANNELS,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        self.is_recording = True
        logger.info("[MIC] Recording started")
        # surface "Listening" in the status pill while mic is open.
        try:
            from assistant.io.status_broadcaster import status, StatusPhase
            status.set(StatusPhase.LISTENING, detail="")
        except Exception:
            pass  # status broadcaster must never break the audio path

    def stop(self) -> np.ndarray:
        """
        Stop recording and return the audio as a 1D float32 numpy array.
        Returns an empty array if nothing was recorded.
        """
        if not self.is_recording:
            logger.warning("Not currently recording")
            return np.array([], dtype="float32")

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.is_recording = False

        if not self._frames:
            logger.warning("No audio frames captured")
            return np.array([], dtype="float32")

        # Concatenate all frames into one 1D array
        audio = np.concatenate(self._frames, axis=0).flatten()
        duration = len(audio) / config.RECORD_SAMPLE_RATE
        logger.info(f"[MIC] Recording stopped — {duration:.1f}s, {len(audio)} samples")
        # clear the Listening pill — THINKING will fire next via the
        # intent classifier path. (LISTENING → IDLE is NOT counted as
        # substantive so the Done flash will not fire on a no-op mic open.)
        try:
            from assistant.io.status_broadcaster import status, StatusPhase
            status.set(StatusPhase.IDLE)
        except Exception:
            pass
        return audio


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert a float32 numpy array to WAV bytes (16-bit PCM, 16 kHz mono)."""
    # Convert float32 [-1, 1] to int16
    int16_audio = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(config.RECORD_CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(config.RECORD_SAMPLE_RATE)
        wf.writeframes(int16_audio.tobytes())

    return buffer.getvalue()


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe audio using the configured backend.

    Args:
        audio: 1D float32 numpy array of recorded audio.

    Returns:
        The transcribed text (empty string on failure).
    """
    if len(audio) == 0:
        return ""

    backend = config.STT_BACKEND

    if backend == "whisper_cpp":
        return _transcribe_whisper_cpp(audio)
    elif backend == "faster_whisper":
        return _transcribe_faster_whisper(audio)
    else:
        logger.error(f"Unknown STT backend: {backend}")
        return ""


def _transcribe_whisper_cpp(audio: np.ndarray) -> str:
    """
    Transcribe via the whisper.cpp HTTP server.
    POST the WAV file to /inference endpoint — same as the C# code did.
    """
    logger.info("Transcribing via whisper.cpp server...")

    wav_bytes = _audio_to_wav_bytes(audio)

    with open("debug_last_recording.wav", "wb") as f:
        f.write(wav_bytes)

    url = config.WHISPER_CPP_URL.rstrip("/") + "/inference"
    try:
        resp = requests.post(
            url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"response_format": "json"},
            timeout=30,
        )
        resp.raise_for_status()

        result = resp.json()
        text = result.get("text", "").strip()
        logger.info(f"whisper.cpp result: \"{text}\"")
        return text

    except requests.ConnectionError:
        logger.error(
            f"Cannot connect to whisper.cpp at {config.WHISPER_CPP_URL}. "
            "Is the server running?"
        )
        return ""
    except Exception as e:
        logger.error(f"whisper.cpp transcription error: {e}")
        return ""


def _transcribe_faster_whisper(audio: np.ndarray) -> str:
    """Transcribe using the faster-whisper library (local, CTranslate2)."""
    logger.info("Transcribing via faster-whisper...")

    model = _get_faster_whisper_model()

    # faster-whisper accepts a numpy float32 array directly
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True,
    )

    text = " ".join(seg.text for seg in segments).strip()
    logger.info(f"faster-whisper result: \"{text}\"")
    return text


# ─── Module-level recorder instance ──────────────────────────────────────────
recorder = Recorder()


# ─── Noise Floor Calibration ──────────────────────────────────────────────────

_THRESHOLD_MULTIPLIER = 4.0   # speech is ~10-50× louder than room noise; 4× catches it
_MIN_THRESHOLD        = 0.008  # floor for very quiet rooms (prevent near-zero thresholds)
_MAX_THRESHOLD        = 0.060  # ceiling for loud environments
_calibrated_threshold = 0.015  # updated by calibrate_noise_floor() at startup


def calibrate_noise_floor(duration: float = 0.8) -> float:
    """
    Sample ambient audio for `duration` seconds and derive a speech-detection
    threshold from the noise floor.

    Uses the 90th-percentile RMS across 50ms chunks (ignores rare spikes) and
    multiplies by _THRESHOLD_MULTIPLIER. Result is clamped to [_MIN, _MAX] and
    stored in _calibrated_threshold for use by record_until_silence().

    Returns the computed threshold.
    """
    global _calibrated_threshold

    chunk_size = int(config.RECORD_SAMPLE_RATE * 0.05)
    rms_values: list[float] = []

    try:
        with sd.InputStream(
            samplerate = config.RECORD_SAMPLE_RATE,
            channels   = config.RECORD_CHANNELS,
            dtype      = "float32",
            blocksize  = chunk_size,
        ) as stream:
            end_ts = _time.time() + duration
            while _time.time() < end_ts:
                chunk, _ = stream.read(chunk_size)
                rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))
    except Exception as e:
        logger.warning(f"[STT] Noise calibration failed: {e} — using default threshold")
        return _calibrated_threshold

    if not rms_values:
        return _calibrated_threshold

    noise_floor = float(np.percentile(rms_values, 90))
    threshold   = noise_floor * _THRESHOLD_MULTIPLIER
    threshold   = max(_MIN_THRESHOLD, min(_MAX_THRESHOLD, threshold))

    _calibrated_threshold = threshold
    logger.info(
        f"[STT] Noise floor: {noise_floor:.4f} RMS  →  "
        f"speech threshold: {threshold:.4f} (×{_THRESHOLD_MULTIPLIER})"
    )
    return threshold


def record_until_silence(
    max_seconds: float = 10.0,
    silence_seconds: float = 1.2,
    silence_threshold: float | None = None,
    min_speech_seconds: float = 0.3,
) -> tuple[np.ndarray, float]:
    """
    Record audio until the user stops talking, then return it.

    Stops when `silence_seconds` of quiet follows at least `min_speech_seconds`
    of speech. Hard cap at `max_seconds`. Opens its own stream (does NOT use the
    module-level recorder) so it can be called from a background thread.

    silence_threshold: RMS below this = silence. Defaults to _calibrated_threshold
                       (set by calibrate_noise_floor() at startup).

    Returns (audio_array, speech_secs) — callers use speech_secs to distinguish
    real speech from Whisper hallucinations on silence.
    """
    threshold  = silence_threshold if silence_threshold is not None else _calibrated_threshold
    chunk_dur  = 0.05
    chunk_size = int(config.RECORD_SAMPLE_RATE * chunk_dur)

    frames: list[np.ndarray] = []
    speech_secs  = 0.0
    silence_ts: float | None = None
    start_ts = _time.time()

    try:
        with sd.InputStream(
            samplerate = config.RECORD_SAMPLE_RATE,
            channels   = config.RECORD_CHANNELS,
            dtype      = "float32",
            blocksize  = chunk_size,
        ) as stream:
            while _time.time() - start_ts < max_seconds:
                chunk, _ = stream.read(chunk_size)
                frames.append(chunk.copy())
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms >= threshold:
                    speech_secs += chunk_dur
                    silence_ts   = None
                elif speech_secs >= min_speech_seconds:
                    if silence_ts is None:
                        silence_ts = _time.time()
                    elif _time.time() - silence_ts >= silence_seconds:
                        break
    except Exception as e:
        logger.warning(f"record_until_silence error: {e}")

    if not frames:
        return np.array([], dtype="float32"), 0.0

    audio    = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / config.RECORD_SAMPLE_RATE
    logger.info(f"[MIC] Follow-up stopped — {duration:.1f}s (speech={speech_secs:.1f}s, threshold={threshold:.4f})")
    return audio, speech_secs
