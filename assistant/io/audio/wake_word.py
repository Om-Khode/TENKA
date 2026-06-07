"""
wake_word.py — openWakeWord-based wake word detection for the Voice Assistant.

Continuously listens for the assistant's wake word (default "TENKA") using
openWakeWord. When detected, triggers the voice assistant pipeline (starts
recording).

openWakeWord is fully open-source, NO API key needed!

SETUP:
  1. pip install openwakeword
  2. Train a custom wake-word model via Google Colab:
     https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb
     Use target phrases that match your assistant name
     (e.g. ["tenka", "ten-ka", "Tenka"]).
  3. Download the .onnx (or .tflite) model file
  4. Place it at: assistant/models/{assistant_name_lower}.onnx
     (default: assistant/models/tenka.onnx)

  Until you have a custom model, the assistant will use
  the built-in "hey jarvis" model as a fallback (no download needed).
"""

import logging
import threading
import time
import numpy as np

from ... import config

logger = logging.getLogger("wake_word")


def _load_wake_model(Model):
    """Decide which wake-word model to load based on current config.

    Returns an openWakeWord Model instance, or None if wake word should be
    disabled. Split out of _listen_loop so it's unit-testable — Model is
    injected (as the openwakeword.Model class) so tests can pass a mock.

    Policy:
      1. Custom model file exists at WAKE_WORD_MODEL_PATH  → use it.
      2. Custom file missing AND WAKE_WORD_BUILTIN is set  → opt-in fallback,
         load built-in with a LOUD warning about threshold mismatch.
      3. Otherwise                                          → return None.
         Wake word stays disabled; push-to-talk (V) still works.

    Previously we silently fell back to hey_jarvis_v0.1 on a missing custom
    model, which combined with the custom-model-tuned 0.02 threshold caused
    false triggers on ambient + TTS audio → infinite "Yes!" feedback loop.
    """
    custom_model = config.WAKE_WORD_MODEL_PATH
    builtin = config.WAKE_WORD_BUILTIN

    if custom_model and custom_model.exists():
        logger.info(f"Loading custom wake word model: {custom_model}")
        return Model(
            wakeword_models=[str(custom_model)],
            inference_framework=config.WAKE_WORD_INFERENCE_FRAMEWORK,
        )

    if custom_model and builtin:
        logger.warning(
            f"Custom wake word model not found at {custom_model}. "
            f"Falling back to built-in '{builtin}' because WAKE_WORD_BUILTIN "
            f"is set. IMPORTANT: the default threshold "
            f"({config.WAKE_WORD_THRESHOLD}) is tuned for a custom model "
            f"and will cause false triggers with a built-in. Raise "
            f"WAKE_WORD_THRESHOLD (try 0.5) to match the built-in's scale."
        )
        return Model(
            wakeword_models=[builtin],
            inference_framework=config.WAKE_WORD_INFERENCE_FRAMEWORK,
        )

    logger.warning(
        f"Wake word disabled — expected custom model at {custom_model} "
        f"but file is missing. Push-to-talk (press 'V') still works. "
        f"To use a built-in wake word instead, add "
        f"WAKE_WORD_BUILTIN=hey_jarvis_v0.1 and an appropriate "
        f"WAKE_WORD_THRESHOLD (try 0.5) to your .env."
    )
    return None


class WakeWordListener:
    """
    Continuously listens to the microphone for the wake word using openWakeWord.
    When detected, calls the provided callback.

    Runs in a background thread so it doesn't block the async loop.
    """

    def __init__(self, on_wake_word):
        """
        Args:
            on_wake_word: Callable (no args) invoked when the wake word is detected.
                          This will be called from a background thread.
        """
        self._on_wake_word = on_wake_word
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False  # Pause during recording/processing

        # SV-1b2: Ring buffer for speaker verification
        # Stores last 3 seconds of raw audio (int16, 16kHz)
        # 3s × 16000 Hz = 48,000 samples = ~96 KB
        from collections import deque
        self._audio_ring: deque = deque(maxlen=48000)

    def start(self):
        """Start listening for the wake word in a background thread."""
        if self._running:
            logger.warning("Wake word listener already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="WakeWordListener",
        )
        self._thread.start()

    def stop(self):
        """Stop the wake word listener."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("Wake word listener stopped")

    def pause(self):
        """Temporarily pause wake word detection (e.g. during recording/TTS)."""
        self._paused = True

    def resume(self):
        """
        Resume wake word detection.
        Clears the audio ring buffer so any TTS audio that played
        during the pause doesn't contaminate speaker verification.
        """
        self._audio_ring.clear()
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._running

    def get_recent_audio(self, seconds: float = 3.0) -> np.ndarray | None:
        """
        SV-1b2: Get the last N seconds of audio from the ring buffer.
        Used by speaker verification to check who said the wake word.

        Args:
            seconds: how many seconds of recent audio to retrieve

        Returns:
            float32 numpy array normalized to [-1, 1], or None if
            not enough audio has accumulated yet.
        """
        samples_needed = int(seconds * 16000)
        if len(self._audio_ring) < samples_needed:
            return None

        # Convert deque → numpy. Deque contains int16 values.
        buf = np.array(list(self._audio_ring), dtype=np.float32)
        buf = buf[-samples_needed:]  # last N seconds
        buf /= 32768.0              # int16 → float32 [-1, 1]
        return buf

    def clear_audio_buffer(self):
        """
        SV-1b2: Clear the ring buffer after TTS playback.
        Prevents the assistant's own voice from being verified as a speaker.
        Called after TTS finishes playing through speakers.
        """
        self._audio_ring.clear()

    def _listen_loop(self):
        """Background thread: read mic audio and check for wake word."""
        try:
            from openwakeword.model import Model
            import sounddevice as sd
        except ImportError as e:
            logger.error(
                f"Missing dependency: {e}. "
                "Install with: pip install openwakeword sounddevice"
            )
            return

        # ── Load the openWakeWord model ──────────────────────────────
        try:
            # openWakeWord requires shared feature extraction models
            # (melspectrogram.onnx, embedding_model.onnx) to be present
            # even when using a custom .onnx model. Download them once.
            import openwakeword
            logger.info("Ensuring openWakeWord base models are downloaded...")
            openwakeword.utils.download_models()

            oww_model = _load_wake_model(Model)
            if oww_model is None:
                return

        except Exception as e:
            logger.error(f"Failed to initialize openWakeWord: {e}")
            return

        # ── Get model names for logging ──────────────────────────────
        model_names = list(oww_model.models.keys())
        logger.info(f"[AUDIO] Wake word listener started — models: {model_names}")
        logger.info(f"   Threshold: {config.WAKE_WORD_THRESHOLD}")
        logger.info("   Say the wake word to activate!")

        # ── Audio settings ───────────────────────────────────────────
        # openWakeWord expects 16kHz 16-bit mono audio in chunks
        sample_rate = 16000
        # 80ms frames recommended by openWakeWord (1280 samples at 16kHz)
        chunk_samples = config.WAKE_WORD_CHUNK_SIZE

        # ── Sliding window for score accumulation ────────────────────
        # Instead of a single threshold, we accumulate scores over a window.
        # Custom-trained openWakeWord models often produce several small spikes
        # (0.02-0.08) in quick succession when the word is spoken.
        # By summing these, we get a more reliable trigger.
        from collections import deque
        WINDOW_SIZE = 15  # ~1.2 seconds at 80ms per frame
        score_window = {name: deque(maxlen=WINDOW_SIZE) for name in model_names}

        # Debug counter
        frame_count = 0
        DEBUG_LOG_INTERVAL = 25  # ~2 seconds

        logger.info(
            f"   Detection mode: sliding window ({WINDOW_SIZE} frames, "
            f"accumulation threshold: {config.WAKE_WORD_THRESHOLD})"
        )

        # ── Main listening loop ──────────────────────────────────────
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_samples,
            ) as stream:
                logger.debug(
                    f"Wake word mic stream opened: rate={sample_rate}, "
                    f"chunk={chunk_samples} samples"
                )

                while self._running:
                    if self._paused:
                        time.sleep(0.1)
                        continue

                    # Read one chunk of audio from the microphone
                    audio_data, overflowed = stream.read(chunk_samples)

                    if overflowed:
                        logger.debug("Audio overflow in wake word stream")

                    # Flatten to 1D int16 array (openWakeWord expects this)
                    audio_frame = audio_data.flatten()

                    # SV-1b2: Accumulate audio into ring buffer for speaker verification
                    self._audio_ring.extend(audio_frame)

                    # Get predictions from all loaded models
                    predictions = oww_model.predict(audio_frame)

                    # Track scores in sliding windows and check accumulated score
                    triggered = False
                    frame_count += 1

                    for model_name, score in predictions.items():
                        # Add score to the sliding window
                        score_window[model_name].append(score)

                        # Sum of recent scores = accumulated confidence
                        accumulated = sum(score_window[model_name])

                        # Debug: log when we see any activity
                        if score > 0.01:
                            logger.info(
                                f"[WAKE] {model_name}: score={score:.4f}, "
                                f"accumulated={accumulated:.4f} / {config.WAKE_WORD_THRESHOLD}"
                            )

                        # Trigger when accumulated score crosses threshold
                        if accumulated >= config.WAKE_WORD_THRESHOLD:
                            logger.info(
                                f"[MIC] Wake word detected! "
                                f"(model: {model_name}, accumulated: {accumulated:.3f})"
                            )

                            # Reset everything
                            oww_model.reset()
                            for sw in score_window.values():
                                sw.clear()

                            # Trigger the callback
                            self._on_wake_word()

                            # Brief cooldown to prevent rapid re-triggers
                            time.sleep(config.WAKE_WORD_COOLDOWN)
                            triggered = True
                            break

                    # Periodic heartbeat (debug level)
                    if not triggered and frame_count % DEBUG_LOG_INTERVAL == 0:
                        accum_str = ", ".join(
                            f"{k}: {sum(v):.4f}"
                            for k, v in score_window.items()
                        )
                        logger.debug(f"[WAKE] Heartbeat — Accumulated: {accum_str}")

        except Exception as e:
            logger.error(f"Wake word listener error: {e}")
        finally:
            logger.info("Wake word listener thread exited")