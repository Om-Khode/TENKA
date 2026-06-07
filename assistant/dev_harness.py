"""
dev_harness.py — Headless HTTP test harness for TENKA pipeline.

Boots the assistant without TTS/STT/audio/Unity, exposes an HTTP API
so Claude Code (or curl) can send text prompts and read responses.

Run:  python -m assistant.dev_harness [--port 8321]

Endpoints:
  POST /chat     {"message": "..."}  → {"response", "intent", "emotion", "latency_ms", "logs"}
  POST /reset    {}                  → {"ok": true}  (start fresh session)
  GET  /health                       → {"status": "ready"}
"""

import asyncio
import json
import logging
import sys
import time

logger = logging.getLogger("assistant.test_harness")


# ─── Response Capture ──────────────────────────────────────────────────────────

_captured: list[dict] = []


async def _mock_speak(text: str, bridge=None, emotion: str = "neutral") -> bool:
    _captured.append({"text": text, "emotion": emotion})
    logger.info(f'[HARNESS-TTS] "{text}" (emotion={emotion})')
    return True


async def _mock_speak_streaming(token_stream, bridge=None, emotion: str = "neutral"):
    full_text = ""
    async for chunk in token_stream:
        full_text += chunk
    _captured.append({"text": full_text, "emotion": emotion})
    logger.info(f'[HARNESS-TTS] "{full_text}" (emotion={emotion}, streamed)')
    return True, full_text


async def _mock_finish_turn(bridge) -> None:
    pass


# ─── Log Capture ───────────────────────────────────────────────────────────────

class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[str] = []
        self.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    def emit(self, record):
        self.records.append(self.format(record))


# ─── Init ──────────────────────────────────────────────────────────────────────

_initialized = False
_pipeline_lock = asyncio.Lock()


async def _init_harness():
    global _initialized
    if _initialized:
        return

    from . import config
    config.SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)

    from .storage.db import init_db
    init_db(config.SANDBOX_DIR / "memory" / "tenka.db")

    from . import memory, personality, preferences, shortcuts, procedures, settings
    memory.init_memory()
    personality.init_personality_db()
    preferences.init_preference_db()
    shortcuts.init_shortcut_db()
    procedures.init_procedure_db()
    settings.init_settings_db()
    config.reload_runtime_settings()

    from . import session as session_mod
    session_mod.init_session_db()
    session_mod.start_session()

    from .core.asyncio_utils import set_main_loop
    set_main_loop(asyncio.get_running_loop())

    from . import telemetry
    telemetry.init_telemetry_db()

    logger.info("Warming embedding model...")
    memory.warm_embed_model()

    # Patch TTS → capture text, skip audio hardware entirely
    import assistant.io.audio.tts as tts_mod
    import assistant.io.audio.streaming as streaming_mod
    import assistant.main as main_mod

    tts_mod.speak = _mock_speak
    tts_mod.is_speaking = lambda: False
    streaming_mod.speak_streaming = _mock_speak_streaming
    streaming_mod.is_speaking = lambda: False
    main_mod._finish_turn = _mock_finish_turn
    main_mod._session_resume_context = ""

    _initialized = True
    logger.info("Test harness initialized (headless, no audio)")


# ─── Chat Processing ──────────────────────────────────────────────────────────

async def _process_chat(message: str) -> dict:
    from .main import process_text_from_queue
    from .io.unity_bridge import NullBridge

    _captured.clear()
    log_capture = _LogCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(log_capture)

    t0 = time.monotonic()
    try:
        await process_text_from_queue("chat", message, NullBridge())
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        root_logger.removeHandler(log_capture)

    latency_ms = int((time.monotonic() - t0) * 1000)

    response_text = ""
    emotion = "neutral"
    if _captured:
        response_text = " ".join(r["text"] for r in _captured if r["text"])
        emotion = _captured[-1].get("emotion", "neutral")

    intent = "unknown"
    for line in log_capture.records:
        if "Intent: " in line:
            intent = line.split("Intent: ", 1)[1].strip()
            break

    return {
        "response": response_text,
        "intent": intent,
        "emotion": emotion,
        "latency_ms": latency_ms,
        "logs": log_capture.records,
    }


# ─── HTTP Server (zero-dependency, raw asyncio) ───────────────────────────────

def _build_http_response(status: str, body: dict) -> bytes:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    header = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    return header.encode("utf-8") + payload


async def _handle_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        if not request_line:
            return

        parts = request_line.decode(errors="replace").strip().split(" ", 2)
        if len(parts) < 2:
            writer.write(_build_http_response("400 Bad Request", {"error": "bad request"}))
            await writer.drain()
            return

        method, path = parts[0], parts[1]

        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            decoded = line.decode(errors="replace").strip()
            if ": " in decoded:
                key, value = decoded.split(": ", 1)
                headers[key.lower()] = value

        content_length = int(headers.get("content-length", "0"))
        body = b""
        if content_length > 0:
            body = await asyncio.wait_for(reader.readexactly(content_length), timeout=5.0)

        if method == "GET" and path == "/health":
            writer.write(_build_http_response("200 OK", {"status": "ready"}))

        elif method == "POST" and path == "/chat":
            try:
                data = json.loads(body)
                msg = data.get("message", "").strip()
            except (json.JSONDecodeError, AttributeError):
                writer.write(_build_http_response("400 Bad Request", {"error": "invalid JSON"}))
                await writer.drain()
                return

            if not msg:
                writer.write(_build_http_response("400 Bad Request", {"error": "empty message"}))
                await writer.drain()
                return

            async with _pipeline_lock:
                result = await _process_chat(msg)
            writer.write(_build_http_response("200 OK", result))

        elif method == "POST" and path == "/reset":
            from . import session as session_mod
            session_mod.end_session()
            session_mod.start_session()
            writer.write(_build_http_response("200 OK", {"ok": True}))

        else:
            writer.write(_build_http_response("404 Not Found", {"error": "not found"}))

        await writer.drain()

    except Exception as e:
        logger.debug(f"[HARNESS] Request error: {e}")
        try:
            writer.write(_build_http_response("500 Internal Server Error", {"error": str(e)}))
            await writer.drain()
        except Exception:
            pass
    finally:
        writer.close()
        await writer.wait_closed()


# ─── Entry Point ───────────────────────────────────────────────────────────────

async def start_server(port: int = 8321):
    await _init_harness()

    server = await asyncio.start_server(_handle_request, "127.0.0.1", port)
    logger.info(f"Serving on http://127.0.0.1:{port}")
    print(f"\nTest harness ready — http://127.0.0.1:{port}")
    print(f"  POST /chat   {{\"message\": \"...\"}}")
    print(f"  POST /reset  {{}}              (new session)")
    print(f"  GET  /health")
    print(f"\nPress Ctrl+C to stop.\n")

    async with server:
        await server.serve_forever()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TENKA headless test harness")
    parser.add_argument("--port", type=int, default=8321, help="HTTP port (default 8321)")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(start_server(args.port))
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
