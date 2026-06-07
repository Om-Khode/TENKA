"""Voice enrollment handlers: enroll_voice, forget_voice (SV-1c/SV-1d)."""

import asyncio
import logging

from .registry import tool_registry

logger = logging.getLogger("actions")


@tool_registry.decorator("enroll_voice")
async def handle_enroll_voice(params: dict, llm_response: str = "", bridge=None, **kwargs) -> str:
    """
    SV-1c: Interactive voice enrollment flow with guided sentences.

    Collects 5 audio samples using prompted sentences for variety.
    Each recording is 5 seconds. Recordings are NOT transcribed —
    used only for ECAPA-TDNN voice embeddings.
    """
    from .. import config
    from ..io.audio import speaker_verify, tts
    from ..io.audio.stt import recorder

    if speaker_verify.is_enrolled():
        return (
            "[surprised] E-eh? I already know your voice! "
            "If you want to re-enroll, say 'forget my voice' first... "
            "or I can add more samples to make it better. Want that?"
        )

    if not speaker_verify._ensure_model():
        return (
            "[worried] Hmm, the speaker verification model isn't loaded. "
            "Check the logs — you might need to install speechbrain."
        )

    _ENROLLMENT_PROMPTS = [
        (f"Say your wake word a few times: Hey {config.ASSISTANT_NAME_DISPLAY}, Hey {config.ASSISTANT_NAME_DISPLAY}, Hey {config.ASSISTANT_NAME_DISPLAY}", 4.0),
        ("The quick brown fox jumps over the lazy dog near the riverbank", 5.0),
        ("I really enjoy listening to music while working on my projects", 5.0),
        ("She sells seashells by the seashore every single Saturday morning", 5.0),
        ("My favorite hobby is exploring new places and trying different food", 5.0),
        ("Today the weather is quite nice and I feel pretty good about everything", 5.0),
    ]

    num_samples = len(_ENROLLMENT_PROMPTS)

    if recorder.is_recording:
        recorder.stop()

    if bridge:
        await tts.speak(
            f"Okay! I need to learn your voice. "
            f"I'll give you {num_samples} prompts — just read each one out loud. "
            f"Ready? Let's go!",
            bridge, emotion="excited"
        )

    samples = []
    for i in range(num_samples):
        await asyncio.sleep(1.0)

        prompt_text, record_seconds = _ENROLLMENT_PROMPTS[i]

        if bridge:
            await tts.speak(
                f"Number {i + 1}. Read this: {prompt_text}",
                bridge, emotion="neutral"
            )

        await asyncio.sleep(0.5)

        recorder.start()
        await asyncio.sleep(record_seconds)
        audio = recorder.stop()

        if audio is not None and len(audio) > 0:
            samples.append(audio)
            logger.info(f"[SV] Enrollment sample {i+1}/{num_samples} captured ({len(audio)} samples)")
        else:
            logger.warning(f"[SV] Enrollment sample {i+1} was empty")

        if i < num_samples - 1 and bridge:
            await tts.speak("Got it!", bridge, emotion="happy")

    if len(samples) < 3:
        return (
            "[worried] Hmm, I didn't get enough good samples. "
            "Can we try again? Say 'enroll my voice' when you're ready."
        )

    result = speaker_verify.enroll_speaker(samples)

    if result["status"] != "enrolled":
        return (
            "[worried] Something went wrong with the enrollment. "
            "Check the logs and try again."
        )

    return (
        f"[happy] All done! I enrolled {result['num_samples']} voice samples. "
        f"From now on, I'll only respond to YOUR voice. "
        f"...N-not that I'd want to talk to anyone else anyway!"
    )


@tool_registry.decorator("forget_voice")
async def handle_forget_voice(params: dict, llm_response: str = "", bridge=None, **kwargs) -> str:
    """Delete the enrolled voiceprint. System returns to fail-open mode."""
    from ..io.audio import speaker_verify

    if not speaker_verify.is_enrolled():
        return (
            "[sarcastic] I don't even know your voice yet. "
            "Can't forget what I never learned, dummy."
        )

    speaker_verify.clear_enrollment()
    return (
        "[sad] Fine... I've forgotten your voice. "
        "We're strangers now, I guess. "
        "Say 'enroll my voice' if you want me to learn it again."
    )
