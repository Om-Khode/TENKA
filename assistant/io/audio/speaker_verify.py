"""
speaker_verify.py — Speaker verification using ECAPA-TDNN (SpeechBrain).

Phase SV-1: Voice gate so the assistant only responds to the enrolled user.

Architecture:
  - ECAPA-TDNN produces 192-dim speaker embeddings
  - Enrollment stores individual embeddings + centroid (averaged)
  - Verification uses cosine similarity with dynamic threshold
  - Dynamic threshold adjusts for SNR (noise) and audio duration
  - "Listen to everyone" toggle for temporary open-mic mode

Three-tier response system:
  Tier 3: No wake word → complete silence (handled elsewhere)
  Tier 2: Wake word + unknown speaker → friendly rejection
  Tier 1: Wake word + verified speaker → full pipeline access

Storage: voiceprint.npz in memory/ directory (numpy arrays, no SQLite)
Pattern: mirrors Phase 4B face recognition (multi-encoding, cosine sim)
"""

import logging
import os
import numpy as np
from pathlib import Path

from ... import config

logger = logging.getLogger("speaker_verify")

# ─── Module State ────────────────────────────────────────────────────────────

_classifier = None              # SpeechBrain EncoderClassifier (lazy loaded)
_enrolled_centroid = None       # np.ndarray (192,) — averaged voiceprint
_enrolled_embeddings: list = [] # list[np.ndarray] — individual samples for best-of-N
# Open-mic state is now config.LISTEN_TO_EVERYONE (persisted via settings_store).
# Accessors below read/write through config + settings_store.
_model_loaded = False           # avoid double-init


# ─── Initialization ─────────────────────────────────────────────────────────

def _patch_speechbrain_lazy_module_windows() -> None:
    """SpeechBrain 1.1.0 has a Windows-incompatible guard in
    `speechbrain.utils.importutils.LazyModule.ensure_module`:

        if importer_frame.filename.endswith("/inspect.py"):
            raise AttributeError()

    The hardcoded forward-slash never matches Windows paths
    (e.g. C:\\Python311\\Lib\\inspect.py). The guard is meant to prevent
    CPython's frame-walking (`inspect.getmodule`, etc.) from triggering real
    imports of optional lazy modules. Without it, frame inspection during
    Playwright's chromium.launch() / asyncio subprocess setup triggers a real
    `import speechbrain.integrations.k2_fsa`, which fails because k2 isn't
    installed → ImportError bubbles up as a Playwright failure.

    We patch ensure_module to also accept backslash paths. Idempotent.
    """
    import sys as _sys
    if not _sys.platform.startswith("win"):
        return
    try:
        from speechbrain.utils import importutils as _iu
    except Exception as e:
        logger.debug(f"[SV] Cannot patch LazyModule (speechbrain not loadable): {e}")
        return
    if getattr(_iu.LazyModule, "_winpath_patched", False):
        return

    import importlib as _importlib
    import inspect as _inspect

    def _patched_ensure_module(self, stacklevel):
        importer_frame = None
        try:
            importer_frame = _inspect.getframeinfo(_sys._getframe(stacklevel + 1))
        except (AttributeError, ValueError):
            pass

        if importer_frame is not None:
            fn = importer_frame.filename.replace("\\", "/")
            if fn.endswith("/inspect.py"):
                raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = _importlib.import_module(self.target)
                else:
                    self.lazy_module = _importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as e:
                raise ImportError(f"Lazy import of {repr(self)} failed") from e
        return self.lazy_module

    _iu.LazyModule.ensure_module = _patched_ensure_module
    _iu.LazyModule._winpath_patched = True
    logger.info("[SV] Patched speechbrain LazyModule for Windows path-separator bug")


def init_speaker_model() -> None:
    """
    Load ECAPA-TDNN from SpeechBrain (pretrained on VoxCeleb).

    Called once at startup from async_main(). Model is ~80MB,
    downloaded on first run and cached in PROJECT_ROOT/models/ecapa_tdnn/.

    Also loads existing voiceprint from disk if available.
    """
    global _classifier, _model_loaded

    if _model_loaded:
        return

    try:
        import torch  # noqa: F401 — verify torch is available
        from speechbrain.inference.speaker import EncoderClassifier
        import sys
        import shutil

        _patch_speechbrain_lazy_module_windows()

        save_dir = config.PROJECT_ROOT / "models" / "ecapa_tdnn"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading ECAPA-TDNN speaker verification model...")

        # On Windows, SpeechBrain tries to create symlinks which fails
        # without admin privileges (WinError 1314). Two-part fix:
        # 1. Use HF cache path as savedir (avoids cache→project symlinks)
        # 2. Monkey-patch os.symlink to shutil.copy2 (catches internal
        #    symlinks like label_encoder.txt → label_encoder.ckpt)
        if sys.platform == "win32":
            try:
                from huggingface_hub import snapshot_download
                cache_dir = snapshot_download(
                    repo_id="speechbrain/spkrec-ecapa-voxceleb",
                )
                logger.info(f"Using HF cache path as savedir: {cache_dir}")
                save_dir = cache_dir
            except Exception as e:
                logger.warning(f"[SV] HF snapshot_download failed: {e}")

            # Patch os.symlink → shutil.copy2 for remaining internal symlinks
            _original_symlink = getattr(os, "symlink", None)

            def _copy_instead_of_symlink(src, dst, *args, **kwargs):
                src_str, dst_str = str(src), str(dst)
                try:
                    if os.path.isdir(src_str):
                        if not os.path.exists(dst_str):
                            shutil.copytree(src_str, dst_str)
                    else:
                        shutil.copy2(src_str, dst_str)
                except shutil.SameFileError:
                    pass
                except Exception as e:
                    logger.debug(f"[SV] copy fallback: {e}")

            os.symlink = _copy_instead_of_symlink
        else:
            _original_symlink = None

        try:
            _classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(save_dir),
                run_opts={"device": "cpu"},
            )
        finally:
            if _original_symlink is not None:
                os.symlink = _original_symlink
        _model_loaded = True
        logger.info("ECAPA-TDNN model loaded successfully")

        # Try loading existing voiceprint
        if _load_voiceprint():
            logger.info(
                f"Voiceprint loaded: {len(_enrolled_embeddings)} samples"
            )
        else:
            logger.info("No voiceprint found — speaker verification inactive until enrollment")

    except ImportError as e:
        logger.error(
            f"Missing dependency for speaker verification: {e}. "
            "Install with: pip install speechbrain torchaudio"
        )
    except Exception as e:
        logger.error(f"Failed to initialize speaker verification: {e}")


def _ensure_model() -> bool:
    """Ensure model is loaded. Returns True if ready."""
    if not _model_loaded:
        init_speaker_model()
    return _classifier is not None


# ─── Embedding Extraction ────────────────────────────────────────────────────

def _trim_silence(audio: np.ndarray, sample_rate: int = 16000,
                   frame_ms: int = 30, energy_threshold: float = 0.01,
                   padding_ms: int = 100) -> np.ndarray:
    """
    Trim silence from audio using simple energy-based VAD.
    
    Keeps only frames where energy exceeds the threshold,
    with padding on both sides to avoid clipping speech edges.
    
    This is critical for wake word verification — a short wake utterance
    (e.g. "Hey TENKA", ~0.8s) inside a 3s ring buffer leaves ECAPA with a
    mostly-silent input that produces a diluted, unreliable embedding.
    
    Args:
        audio: float32 numpy array
        sample_rate: audio sample rate
        frame_ms: frame size in milliseconds
        energy_threshold: RMS energy threshold (fraction of peak)
        padding_ms: padding to keep around speech regions
    
    Returns:
        Trimmed audio (float32). Returns original if trimming would
        leave less than 0.3s of audio.
    """
    if len(audio) == 0:
        return audio

    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(audio) // frame_size
    if num_frames < 2:
        return audio

    # Compute per-frame RMS energy
    frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Dynamic threshold: use a fraction of peak RMS
    peak_rms = np.max(rms)
    if peak_rms < 1e-6:
        return audio  # All silence
    threshold = peak_rms * energy_threshold

    # Find frames with speech
    speech_mask = rms > threshold
    speech_indices = np.where(speech_mask)[0]

    if len(speech_indices) == 0:
        return audio  # No speech detected, return as-is

    # Get start and end with padding
    pad_frames = int(padding_ms / frame_ms)
    start_frame = max(0, speech_indices[0] - pad_frames)
    end_frame = min(num_frames, speech_indices[-1] + pad_frames + 1)

    start_sample = start_frame * frame_size
    end_sample = min(end_frame * frame_size, len(audio))

    trimmed = audio[start_sample:end_sample]

    # Don't return if too short (< 0.3s)
    min_samples = int(0.3 * sample_rate)
    if len(trimmed) < min_samples:
        return audio

    logger.debug(
        f"[SV] Trimmed audio: {len(audio)/sample_rate:.1f}s → "
        f"{len(trimmed)/sample_rate:.1f}s "
        f"(kept {len(speech_indices)}/{num_frames} frames)"
    )

    return trimmed


def _extract_embedding(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract 192-dim ECAPA-TDNN embedding from raw audio.

    Args:
        audio: numpy array (int16 or float32), mono, 16kHz
        sample_rate: sample rate (must be 16000)

    Returns:
        L2-normalized np.ndarray of shape (192,)
    """
    import torch

    # Convert int16 → float32 if needed
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Ensure 1D
    if audio.ndim > 1:
        audio = audio.flatten()

    # NOTE: VAD trimming was tried here but caused embedding mismatch
    # between enrollment and verification audio. Removed.
    # _trim_silence() is kept as a utility for future wake word use.

    # Normalize amplitude
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # Convert to torch tensor with batch dimension: (1, num_samples)
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    # Extract embedding
    with torch.no_grad():
        embedding = _classifier.encode_batch(waveform)

    # Squeeze to 1D numpy array
    emb = embedding.squeeze().cpu().numpy()

    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb


# ─── Similarity ──────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(a, b))


def _best_of_n_score(query: np.ndarray, stored: list) -> float:
    """Highest cosine similarity across all stored enrollment embeddings."""
    if not stored:
        return 0.0
    return max(_cosine_similarity(query, s) for s in stored)


# ─── Dynamic Threshold ───────────────────────────────────────────────────────

def _estimate_snr(audio: np.ndarray) -> float:
    """
    Simple SNR estimation from audio energy distribution.

    Splits audio into 50ms frames, sorts by energy,
    compares top-20% (signal) vs bottom-20% (noise).

    Returns estimated SNR in dB.
    """
    # Ensure float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Split into 50ms frames (800 samples at 16kHz)
    frame_size = 800
    num_frames = len(audio) // frame_size
    if num_frames < 5:
        return 20.0  # Default moderate SNR for very short audio

    frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
    energies = np.mean(frames ** 2, axis=1)

    # Sort energies
    sorted_energies = np.sort(energies)

    # Bottom 20% = noise estimate, top 20% = signal estimate
    n_low = max(1, num_frames // 5)
    n_high = max(1, num_frames // 5)

    noise_energy = np.mean(sorted_energies[:n_low])
    signal_energy = np.mean(sorted_energies[-n_high:])

    # Avoid log(0)
    if noise_energy < 1e-10:
        return 40.0  # Very clean
    if signal_energy < 1e-10:
        return 0.0   # No signal

    snr_db = 10 * np.log10(signal_energy / noise_energy)
    return float(snr_db)


def _get_effective_threshold(audio: np.ndarray, offset: float = 0.0) -> float:
    """
    Compute dynamic threshold based on audio quality.

    Base: config.SPEAKER_VERIFY_THRESHOLD (default 0.65)
    + offset (e.g., -0.05 for wake word audio)

    SNR adjustments:
        > 25 dB (clean):      no adjustment
        15-25 dB (moderate):  -0.03
        < 15 dB (noisy):     -0.06

    Duration adjustment:
        < 1.5 seconds:        -0.04

    Floor: config.SPEAKER_VERIFY_THRESHOLD_FLOOR (default 0.50)
    """
    threshold = config.SPEAKER_VERIFY_THRESHOLD + offset

    # SNR-based adjustment
    snr = _estimate_snr(audio)
    if snr < 15.0:
        threshold -= 0.06
        logger.debug(f"[SV] Noisy environment (SNR={snr:.1f}dB), threshold -0.06")
    elif snr < 25.0:
        threshold -= 0.03
        logger.debug(f"[SV] Moderate noise (SNR={snr:.1f}dB), threshold -0.03")

    # Duration-based adjustment
    duration = len(audio) / 16000.0
    if duration < 1.5:
        threshold -= 0.04
        logger.debug(f"[SV] Short audio ({duration:.1f}s), threshold -0.04")

    # Apply floor
    floor = config.SPEAKER_VERIFY_THRESHOLD_FLOOR
    if threshold < floor:
        threshold = floor

    return threshold


# ─── Verification ────────────────────────────────────────────────────────────

def verify(audio: np.ndarray, sample_rate: int = 16000,
           threshold_offset: float = 0.0) -> dict:
    """
    Check if audio belongs to the enrolled speaker.

    Args:
        audio: raw audio (int16 or float32, 16kHz mono)
        sample_rate: audio sample rate (must be 16000)
        threshold_offset: added to base threshold.
                          Use -0.05 for wake word audio (shorter/fixed-phrase)

    Returns:
        {
            "is_owner": bool,
            "score": float,       # cosine similarity (0.0 - 1.0)
            "threshold": float,   # effective threshold used
            "method": str,        # "centroid" | "best_of_n" | "open" | "not_enrolled" | "too_short"
        }
    """
    # Listen to everyone mode — bypass all checks (persisted via settings_store)
    if config.LISTEN_TO_EVERYONE:
        return {"is_owner": True, "score": 1.0, "threshold": 0.0, "method": "open"}

    # Not enrolled — fail-open (system usable before enrollment)
    if not is_enrolled():
        return {"is_owner": True, "score": 1.0, "threshold": 0.0, "method": "not_enrolled"}

    # Model not loaded
    if not _ensure_model():
        logger.warning("[SV] Model not loaded — failing open")
        return {"is_owner": True, "score": 1.0, "threshold": 0.0, "method": "not_enrolled"}

    # Check minimum audio duration
    if audio.dtype == np.int16:
        duration = len(audio) / 16000.0
    else:
        duration = len(audio) / 16000.0

    if duration < config.SPEAKER_MIN_AUDIO_SECONDS:
        logger.debug(f"[SV] Audio too short ({duration:.2f}s) — failing open")
        return {"is_owner": True, "score": 0.0, "threshold": 0.0, "method": "too_short"}

    # Extract embedding from input audio
    try:
        query_emb = _extract_embedding(audio, sample_rate)
    except Exception as e:
        logger.error(f"[SV] Embedding extraction failed: {e} — failing open")
        return {"is_owner": True, "score": 0.0, "threshold": 0.0, "method": "error"}

    # Compute dynamic threshold
    threshold = _get_effective_threshold(audio, offset=threshold_offset)

    # Fast path: compare against centroid
    centroid_score = _cosine_similarity(query_emb, _enrolled_centroid)

    if centroid_score >= threshold:
        logger.info(
            f"[SV] Owner verified via centroid "
            f"(score={centroid_score:.3f}, threshold={threshold:.3f})"
        )
        return {
            "is_owner": True,
            "score": centroid_score,
            "threshold": threshold,
            "method": "centroid",
        }

    # Close call — try best-of-N against individual embeddings
    # This catches edge cases where the centroid misses (sick voice, whispering)
    margin = 0.05
    if centroid_score >= threshold - margin and len(_enrolled_embeddings) > 1:
        best_score = _best_of_n_score(query_emb, _enrolled_embeddings)

        if best_score >= threshold:
            logger.info(
                f"[SV] Owner verified via best-of-N "
                f"(centroid={centroid_score:.3f}, best={best_score:.3f}, "
                f"threshold={threshold:.3f})"
            )
            return {
                "is_owner": True,
                "score": best_score,
                "threshold": threshold,
                "method": "best_of_n",
            }
        else:
            logger.info(
                f"[SV] Rejected — close call but failed "
                f"(centroid={centroid_score:.3f}, best={best_score:.3f}, "
                f"threshold={threshold:.3f})"
            )
            return {
                "is_owner": False,
                "score": best_score,
                "threshold": threshold,
                "method": "best_of_n",
            }

    # Clear rejection
    logger.info(
        f"[SV] Rejected unknown speaker "
        f"(score={centroid_score:.3f}, threshold={threshold:.3f})"
    )
    return {
        "is_owner": False,
        "score": centroid_score,
        "threshold": threshold,
        "method": "centroid",
    }


# ─── Enrollment ──────────────────────────────────────────────────────────────

def enroll_speaker(audio_samples: list, sample_rate: int = 16000) -> dict:
    """
    Enroll the owner's voice from multiple audio samples.

    Args:
        audio_samples: list of numpy arrays (16kHz mono, int16 or float32).
                       Minimum 3 samples recommended.
        sample_rate: audio sample rate (must be 16000)

    Returns:
        {"status": "enrolled", "num_samples": int}
    """
    global _enrolled_centroid, _enrolled_embeddings

    if not _ensure_model():
        return {"status": "error", "num_samples": 0}

    embeddings = []
    for i, audio in enumerate(audio_samples):
        try:
            emb = _extract_embedding(audio, sample_rate)
            embeddings.append(emb)
            logger.info(f"[SV] Enrollment sample {i+1}/{len(audio_samples)} processed")
        except Exception as e:
            logger.warning(f"[SV] Enrollment sample {i+1} failed: {e}")

    if len(embeddings) < 1:
        return {"status": "error", "num_samples": 0}

    # Cap at maximum
    max_enroll = config.SPEAKER_MAX_ENROLLMENTS
    if len(embeddings) > max_enroll:
        embeddings = embeddings[:max_enroll]

    # Store individual embeddings
    _enrolled_embeddings = embeddings

    # Compute centroid (mean, L2-normalized)
    centroid = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    _enrolled_centroid = centroid

    # Save to disk
    _save_voiceprint()

    logger.info(
        f"[SV] Speaker enrolled with {len(embeddings)} samples"
    )
    return {"status": "enrolled", "num_samples": len(embeddings)}


def add_enrollment_sample(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Add a single additional sample to existing enrollment.
    Re-computes centroid. Caps at SPEAKER_MAX_ENROLLMENTS.

    Returns:
        {"status": "added" | "error", "num_samples": int}
    """
    global _enrolled_centroid, _enrolled_embeddings

    if not _ensure_model():
        return {"status": "error", "num_samples": len(_enrolled_embeddings)}

    if not is_enrolled():
        return {"status": "error", "num_samples": 0}

    try:
        emb = _extract_embedding(audio, sample_rate)
    except Exception as e:
        logger.warning(f"[SV] Add sample failed: {e}")
        return {"status": "error", "num_samples": len(_enrolled_embeddings)}

    # Cap at maximum — drop oldest if full
    max_enroll = config.SPEAKER_MAX_ENROLLMENTS
    if len(_enrolled_embeddings) >= max_enroll:
        _enrolled_embeddings.pop(0)
        logger.debug(f"[SV] Dropped oldest enrollment (cap={max_enroll})")

    _enrolled_embeddings.append(emb)

    # Re-compute centroid
    centroid = np.mean(np.stack(_enrolled_embeddings), axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    _enrolled_centroid = centroid

    _save_voiceprint()

    logger.info(f"[SV] Added enrollment sample (total: {len(_enrolled_embeddings)})")
    return {"status": "added", "num_samples": len(_enrolled_embeddings)}


def clear_enrollment() -> None:
    """Delete voiceprint and clear enrolled data."""
    global _enrolled_centroid, _enrolled_embeddings

    _enrolled_centroid = None
    _enrolled_embeddings = []

    vp_path = config.SPEAKER_VOICEPRINT_PATH
    try:
        if vp_path.exists():
            vp_path.unlink()
            logger.info(f"[SV] Voiceprint deleted: {vp_path}")
    except Exception as e:
        logger.warning(f"[SV] Failed to delete voiceprint: {e}")


# ─── Listen to Everyone Toggle ───────────────────────────────────────────────

def set_listen_to_everyone(enabled: bool) -> None:
    """Toggle open-mic mode. Persists across restarts via settings_store."""
    from ... import settings  # local import — settings initialized at startup
    settings.set("listen_to_everyone", bool(enabled), source="user")
    config.reload_runtime_settings()
    logger.info(f"[SV] Listen to everyone: {'ON' if enabled else 'OFF'} (persisted)")


def is_listen_to_everyone() -> bool:
    """Check current open-mic state."""
    return bool(config.LISTEN_TO_EVERYONE)


# ─── Persistence ─────────────────────────────────────────────────────────────

def _save_voiceprint() -> None:
    """
    Save voiceprint to disk as .npz file.
    Stores centroid + individual embeddings.
    """
    global _enrolled_centroid, _enrolled_embeddings

    vp_path = config.SPEAKER_VOICEPRINT_PATH
    try:
        vp_path.parent.mkdir(parents=True, exist_ok=True)

        # Pack embeddings into a dict for np.savez
        save_dict = {"centroid": _enrolled_centroid}
        for i, emb in enumerate(_enrolled_embeddings):
            save_dict[f"emb_{i}"] = emb

        np.savez(str(vp_path), **save_dict)
        logger.info(f"[SV] Voiceprint saved: {vp_path}")

    except Exception as e:
        logger.error(f"[SV] Failed to save voiceprint: {e}")


def _load_voiceprint() -> bool:
    """
    Load voiceprint from disk. Called during init.
    Returns True if loaded successfully.
    """
    global _enrolled_centroid, _enrolled_embeddings

    vp_path = config.SPEAKER_VOICEPRINT_PATH
    if not vp_path.exists():
        return False

    try:
        data = np.load(str(vp_path))

        if "centroid" not in data:
            logger.warning("[SV] Voiceprint file missing centroid — ignoring")
            return False

        _enrolled_centroid = data["centroid"]

        # Load individual embeddings
        _enrolled_embeddings = []
        i = 0
        while f"emb_{i}" in data:
            _enrolled_embeddings.append(data[f"emb_{i}"])
            i += 1

        logger.info(
            f"[SV] Voiceprint loaded: centroid + {len(_enrolled_embeddings)} embeddings"
        )
        return True

    except Exception as e:
        logger.error(f"[SV] Failed to load voiceprint: {e}")
        return False


def is_enrolled() -> bool:
    """Check if a voiceprint has been enrolled."""
    return _enrolled_centroid is not None and len(_enrolled_embeddings) > 0