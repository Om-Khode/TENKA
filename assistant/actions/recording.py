"""Recording handlers: start, stop, get, summarize recording sessions."""

import logging

from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")


@tool_registry.decorator("start_recording")
async def handle_start_recording(params: dict, llm_response: str, bridge) -> str:
    from .. import recording
    if recording.is_active():
        return personality_say("recording_already_active")
    try:
        session_id = recording.start_session()
        return personality_say("recording_started")
    except Exception as e:
        return f"Failed to start recording session: {e}"


@tool_registry.decorator("stop_recording")
async def handle_stop_recording(params: dict, llm_response: str, bridge) -> str:
    from .. import recording
    if not recording.is_active():
        return personality_say("recording_not_active")
    try:
        result = recording.stop_session()
        chunk_count = result["chunk_count"]
        duration = result["duration_seconds"]
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        base_msg = f"Recording stopped. {chunk_count} chunk{'s' if chunk_count != 1 else ''} saved, total duration {duration_str}."

        if chunk_count >= recording.SUMMARY_THRESHOLD:
            recording._pending_summary = result
            return base_msg + " That was a fairly long session — would you like me to summarize it?"
        else:
            recording._pending_summary = None
            return base_msg
    except Exception as e:
        return f"Failed to stop recording session: {e}"


@tool_registry.decorator("get_recording")
async def handle_get_recording(params: dict, llm_response: str, bridge) -> str:
    from .. import memory as mem
    from ..llm.contracts import ask_for_synthesis

    session_id = params.get("session_id", "latest")

    if session_id == "latest":
        sessions = mem.list_sessions(limit=1)
        if not sessions:
            return "No recording sessions found."
        session_id = sessions[0]["session_id"]

    chunks = mem.get_session_transcript(session_id)
    if not chunks:
        return f"No transcript found for session '{session_id}'."

    full_text = "\n".join(c["transcript"] for c in chunks)
    summary_prompt = (
        f"The following is a transcript of a voice recording session "
        f"({len(chunks)} chunks). Give a single short spoken sentence "
        f"(under 30 words) describing what the session was about, "
        f"suitable for text-to-speech:\n\n{full_text}"
    )
    spoken_summary = await ask_for_synthesis(summary_prompt)
    if spoken_summary == "__LLM_UNAVAILABLE__":
        spoken_summary = f"Found session with {len(chunks)} chunks."

    return spoken_summary


@tool_registry.decorator("summarize_recording")
async def handle_summarize_recording(params: dict, llm_response: str, bridge) -> str:
    from .. import memory as mem
    from ..llm.contracts import ask_for_synthesis

    session_id = params.get("session_id", "latest")

    if session_id == "latest":
        sessions = mem.list_sessions(limit=1)
        if not sessions:
            return "No recording sessions found to summarize."
        session_id = sessions[0]["session_id"]

    chunks = mem.get_session_transcript(session_id)
    if not chunks:
        return f"No transcript found for session '{session_id}'."

    full_text = "\n".join(c["transcript"] for c in chunks)
    duration_info = f"{len(chunks)} chunks"

    summary_prompt = (
        f"The following is a transcript of a voice recording session ({duration_info}). "
        f"Write a concise bullet-point summary of the key points, topics mentioned, "
        f"and any important details like names, numbers, or dates:\n\n{full_text}"
    )

    summary = await ask_for_synthesis(summary_prompt)
    if summary == "__LLM_UNAVAILABLE__":
        return f"Couldn't summarize — LLM unavailable. Session had {len(chunks)} chunks."

    return f"Summary of session '{session_id}':\n{summary}"
