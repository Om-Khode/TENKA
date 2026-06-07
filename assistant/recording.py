"""
recording.py — Long-form recording session manager for TENKA.

VAD-based chunking: audio is recorded continuously in a background thread.
Whenever a 2-second silence gap is detected OR 60 seconds of audio accumulate,
the current buffer is transcribed and saved as a chunk to SQLite.

Usage:
    from . import recording

    # Start a session (returns session_id string)
    session_id = recording.start_session()

    # Stop the session (returns summary dict)
    result = recording.stop_session()
    # result = {
    #     "session_id": str,
    #     "chunk_count": int,
    #     "duration_seconds": float,
    #     "chunks": [{"chunk_index": int, "transcript": str}, ...]
    # }

    # Check if a session is active
    recording.is_active()  # bool
"""

import logging
import threading
import time
import queue
import numpy as np
from datetime import datetime

from . import memory

logger = logging.getLogger("recording")

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16000   # Hz — matches faster-whisper expectation
CHUNK_SECONDS      = 0.1     # Audio read chunk size (100ms slices)
VAD_SILENCE_GAP    = 2.0     # Seconds of silence that ends a speech segment
MAX_CHUNK_SECONDS  = 60.0    # Force-save after this many seconds regardless of silence
SUMMARY_THRESHOLD  = 5       # Offer summary if session has this many chunks or more

# ─── Stop-command detection ──────────────────────────────────────────────────
# When the recording worker transcribes a stop phrase, it signals the main loop
# instead of saving it as a chunk.

_STOP_PHRASES = frozenset({
    "stop recording", "stop the recording",
    "end recording", "end the recording",
    "finish recording", "finish the recording",
})

_voice_stop_event: threading.Event = threading.Event()


def voice_stop_requested() -> bool:
    """Check and clear the voice-stop flag (polled by the main loop)."""
    if _voice_stop_event.is_set():
        _voice_stop_event.clear()
        return True
    return False


# ─── State ────────────────────────────────────────────────────────────────────

_session_id: str | None = None
_session_start: float | None = None
_chunk_index: int = 0
_active: bool = False
_stop_event: threading.Event = threading.Event()
_worker_thread: threading.Thread | None = None
_suppress_until: float = 0.0   # Timestamp until which chunk flushing is suppressed

# ─── VAD helper ───────────────────────────────────────────────────────────────

def _is_silence(audio_chunk: np.ndarray, threshold: float) -> bool:
    """Return True if the audio chunk is below the silence energy threshold."""
    if len(audio_chunk) == 0:
        return True
    normalized = audio_chunk.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(normalized ** 2)))
    return rms < threshold


# ─── Public API ───────────────────────────────────────────────────────────────


def is_active() -> bool:
    """Return True if a recording session is currently running."""
    return _active


def suppress_for(seconds: float = 3.0) -> None:
    """
    Suppress chunk saving for the given number of seconds.
    Call this when wake word is detected during an active session
    to prevent wake word audio from bleeding into the transcript.
    """
    global _suppress_until
    if _active:
        _suppress_until = time.time() + seconds
        logger.debug(f"[RECORDING] Suppressing chunk flush for {seconds}s")


def start_session() -> str:
    """
    Start a new recording session in a background thread.

    Returns:
        session_id string (e.g. 'recording_20240315_143022').
    Raises:
        RuntimeError if a session is already active.
    """
    global _session_id, _session_start, _chunk_index, _active, _stop_event, _worker_thread

    if _active:
        raise RuntimeError("A recording session is already active. Call stop_session() first.")

    _session_id = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _session_start = time.time()
    _chunk_index = 0
    _active = True
    _stop_event = threading.Event()

    _worker_thread = threading.Thread(
        target=_recording_worker,
        args=(_session_id, _stop_event),
        daemon=True,
        name="recording-worker",
    )
    _worker_thread.start()

    logger.info(f"[RECORDING] Session started: {_session_id}")
    return _session_id


def stop_session() -> dict:
    """
    Stop the current recording session and return a summary dict.

    Returns:
        {
            "session_id": str,
            "chunk_count": int,
            "duration_seconds": float,
            "chunks": [{"chunk_index": int, "transcript": str}, ...]
        }
    Raises:
        RuntimeError if no session is active.
    """
    global _active

    if not _active:
        raise RuntimeError("No recording session is active.")

    logger.info("[RECORDING] Stopping session...")
    _stop_event.set()

    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=10.0)

    _active = False
    session_id = _session_id
    duration = time.time() - (_session_start or time.time())

    # Fetch saved chunks from DB
    chunks = memory.get_session_transcript(session_id)
    result = {
        "session_id": session_id,
        "chunk_count": len(chunks),
        "duration_seconds": round(duration, 1),
        "chunks": [{"chunk_index": c["chunk_index"], "transcript": c["transcript"]} for c in chunks],
    }

    logger.info(f"[RECORDING] Session stopped: {session_id} — {len(chunks)} chunks, {duration:.1f}s")
    return result


# ─── Background Worker ────────────────────────────────────────────────────────


def _calibrate_noise_floor(stream, sample_rate: int, duration: float = 1.5) -> float:
    """
    Record duration seconds of ambient audio and return a silence threshold
    set at 1.5x the measured RMS noise floor.
    Minimum threshold is 0.005 to avoid false triggers in dead silence.
    """
    chunk_size = int(sample_rate * 0.1)
    rms_values = []
    samples_needed = int(duration / 0.1)

    for _ in range(samples_needed):
        block, _ = stream.read(chunk_size)
        audio_np = block.flatten().astype(np.float32)
        normalized = audio_np / 32768.0
        rms = float(np.sqrt(np.mean(normalized ** 2)))
        rms_values.append(rms)

    noise_floor = float(np.mean(rms_values))
    threshold = max(noise_floor * 1.5, 0.005)
    logger.info(f"[RECORDING] Noise floor calibrated: rms={noise_floor:.4f}, threshold={threshold:.4f}")
    return threshold


def _recording_worker(session_id: str, stop_event: threading.Event):
    """
    Background thread: continuously reads mic audio, applies VAD,
    and saves chunks when silence is detected or max duration is reached.
    """
    global _chunk_index

    try:
        import sounddevice as sd
    except ImportError:
        logger.error("[RECORDING] sounddevice not installed — pip install sounddevice")
        return

    try:
        from .io.audio import stt as stt_module
        transcribe_func = stt_module.transcribe
    except Exception as e:
        logger.error(f"[RECORDING] Could not import stt.transcribe: {e}")
        return

    # Audio accumulation buffer (raw int16 samples)
    speech_buffer: list[np.ndarray] = []
    silence_duration: float = 0.0
    chunk_duration: float = 0.0

    logger.info(f"[RECORDING] Worker started for session {session_id}")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
        ) as stream:
            # Calibrate noise floor from first 1.5s of ambient audio
            vad_threshold = _calibrate_noise_floor(stream, SAMPLE_RATE)
            logger.info(f"[RECORDING] VAD ready — silence threshold={vad_threshold:.4f}")

            while not stop_event.is_set():
                audio_block, _ = stream.read(int(SAMPLE_RATE * CHUNK_SECONDS))
                audio_np = audio_block.flatten()

                if _is_silence(audio_np, vad_threshold):
                    silence_duration += CHUNK_SECONDS
                    # Still accumulate silence into buffer (natural speech has gaps)
                    if speech_buffer:
                        speech_buffer.append(audio_np)
                else:
                    silence_duration = 0.0
                    speech_buffer.append(audio_np)

                chunk_duration += CHUNK_SECONDS

                # Decide whether to flush: VAD silence gap OR max chunk length
                should_flush = (
                    silence_duration >= VAD_SILENCE_GAP and len(speech_buffer) > 0
                ) or (
                    chunk_duration >= MAX_CHUNK_SECONDS and len(speech_buffer) > 0
                )

                if should_flush:
                    if time.time() < _suppress_until:
                        # Wake word suppression active — discard this buffer silently
                        logger.debug("[RECORDING] Chunk suppressed (wake word window)")
                        speech_buffer = []
                        silence_duration = 0.0
                        chunk_duration = 0.0
                    else:
                        _flush_chunk(session_id, speech_buffer, transcribe_func)
                        speech_buffer = []
                        silence_duration = 0.0
                        chunk_duration = 0.0

    except Exception as e:
        logger.error(f"[RECORDING] Worker error: {e}", exc_info=True)

    # Flush any remaining audio when stop is requested
    if speech_buffer:
        logger.info("[RECORDING] Flushing final buffer on stop...")
        _flush_chunk(session_id, speech_buffer, transcribe_func)

    logger.info(f"[RECORDING] Worker exited for session {session_id}")


def _flush_chunk(session_id: str, buffers: list[np.ndarray], transcribe_func):
    """Concatenate buffered audio, transcribe it, and save to DB."""
    global _chunk_index

    audio = np.concatenate(buffers).astype(np.float32) / 32768.0  # int16 → float32 [-1, 1]

    try:
        transcript = transcribe_func(audio)
    except Exception as e:
        logger.warning(f"[RECORDING] Transcription failed for chunk {_chunk_index}: {e}")
        return

    if not transcript or not transcript.strip():
        logger.debug(f"[RECORDING] Empty transcript for chunk {_chunk_index} — skipping save")
        return

    cleaned = transcript.strip()
    if cleaned.lower().rstrip(".!?,") in _STOP_PHRASES:
        logger.info(f"[RECORDING] Detected stop command in transcript: {cleaned!r}")
        _voice_stop_event.set()
        return

    memory.save_chunk(session_id, _chunk_index, cleaned)
    logger.info(f"[RECORDING] Chunk {_chunk_index} saved ({len(transcript)} chars): {transcript[:60]}...")
    _chunk_index += 1
