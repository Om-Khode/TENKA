"""
unity_bridge.py — TCP socket bridge between Python and Unity.

Two servers run concurrently:
  1. COMMAND server (port 7777): Python sends JSON commands TO Unity.
     Unity connects as a client and receives commands like:
       {"action": "set_expression", "value": "happy"}
       {"action": "play_animation", "name": "thinking"}
       {"action": "set_talking", "value": true}
       {"action": "show_subtitle", "text": "Hello!"}

  2. EVENT server (port 7778): Python receives JSON events FROM Unity.
     Unity connects as a client and sends events like:
       {"event": "start_listening"}
       {"event": "stop_listening"}
       {"event": "avatar_clicked"}

Protocol: Each message is framed as:
  [4 bytes: big-endian uint32 length] + [N bytes: UTF-8 JSON]

This allows reliable parsing even if TCP delivers partial data.
"""

import asyncio
import json
import struct
import logging

from .. import config

logger = logging.getLogger("bridge")


class UnityBridge:
    """Manages TCP communication between the Python assistant and Unity."""

    def __init__(self):
        # Connected Unity client for sending commands (port 7777)
        self._command_writer: asyncio.StreamWriter | None = None
        self._command_lock = asyncio.Lock()

        # Callback for when Unity sends us an event (port 7778)
        self._event_callback = None

        # Server references (so we can close them on shutdown)
        self._command_server: asyncio.Server | None = None
        self._event_server: asyncio.Server | None = None

        # Track connection state
        self.unity_connected = False

    # ─── Public API ───────────────────────────────────────────────────

    async def start(self, event_callback=None):
        """
        Start both TCP servers. Call this once at startup.

        Args:
            event_callback: async function(event_dict) called when Unity sends an event.
        """
        self._event_callback = event_callback

        # Start the COMMAND server (Python → Unity) on port 7777
        try:
            self._command_server = await asyncio.start_server(
                self._handle_command_client,
                "127.0.0.1",
                config.UNITY_COMMAND_PORT,
            )
            logger.info(f"Command server listening on 127.0.0.1:{config.UNITY_COMMAND_PORT}")
        except OSError as e:
            if e.errno == 10048:
                msg = (
                    f"Port {config.UNITY_COMMAND_PORT} is already in use. "
                    "A previous assistant process may still be running.\n"
                    f"Run: netstat -ano | findstr :{config.UNITY_COMMAND_PORT}  then: taskkill /PID <pid> /F"
                )
                logger.error(msg)
                raise RuntimeError(msg) from e
            raise

        # Start the EVENT server (Unity → Python) on port 7778
        self._event_server = await asyncio.start_server(
            self._handle_event_client,
            "127.0.0.1",
            config.UNITY_EVENT_PORT,
        )
        logger.info(f"Event server listening on 127.0.0.1:{config.UNITY_EVENT_PORT}")

    async def send_command(self, action: str, **kwargs):
        """
        Send a JSON command to Unity.

        Example:
            await bridge.send_command("set_expression", value="happy")
            await bridge.send_command("play_animation", name="wave")
            await bridge.send_command("show_subtitle", text="Hello!")
        """
        message = {"action": action, **kwargs}
        await self._send_to_unity(message)

    async def send_thought(self, state: str, text: str = ""):
        """Send thought bubble command to Unity. state: 'thinking' | 'done'"""
        cmd = {"action": "show_thought", "state": state}
        if text:
            cmd["text"] = text
        await self._send_to_unity(cmd)

    async def send_keyboard(self, visible: bool):
        """Show or hide the keyboard prop in Unity."""
        await self._send_to_unity({"action": "show_keyboard", "value": str(visible).lower()})

    async def send_avatar_config(self):
        """Send avatar hide/peek configuration to Unity."""
        await self.send_command(
            "set_avatar_hide_config",
            peek_duration=config.AVATAR_PEEK_DURATION,
            peek_interval_min=config.AVATAR_PEEK_INTERVAL_MIN,
            peek_interval_max=config.AVATAR_PEEK_INTERVAL_MAX,
            sliver=config.AVATAR_HIDDEN_SLIVER,
            lerp_speed=config.AVATAR_LERP_SPEED,
        )

    async def stop(self):
        """Shut down both servers and close connections."""
        if self._command_server:
            self._command_server.close()
            await self._command_server.wait_closed()
        if self._event_server:
            self._event_server.close()
            await self._event_server.wait_closed()
        if self._command_writer:
            self._command_writer.close()
        logger.info("Bridge servers stopped")

    # ─── Command Server (Python → Unity) ─────────────────────────────

    async def _handle_command_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Called when Unity connects to the command port (7777)."""
        addr = writer.get_extra_info("peername")
        logger.info(f"Unity connected to command port from {addr}")

        # Store the writer so send_command() can use it
        async with self._command_lock:
            # Close previous connection if any
            if self._command_writer:
                try:
                    self._command_writer.close()
                except Exception:
                    pass
            self._command_writer = writer
            self.unity_connected = True

        # Send initial config to Unity
        await self.send_avatar_config()

        try:
            # Keep the connection alive — Unity might also send pings/acks
            while True:
                data = await reader.read(1024)
                if not data:
                    break  # Unity disconnected
        except (asyncio.CancelledError, ConnectionError):
            pass
        finally:
            logger.info("Unity disconnected from command port")
            async with self._command_lock:
                self._command_writer = None
                self.unity_connected = False

    async def _send_to_unity(self, message: dict):
        """Send a length-prefixed JSON message to Unity."""
        async with self._command_lock:
            writer = self._command_writer
            if writer is None:
                # Unity not connected — that's okay, just log and skip
                logger.debug(f"Unity not connected, skipping command: {message}")
                return

        try:
            json_bytes = json.dumps(message).encode("utf-8")
            # 4-byte big-endian length prefix
            header = struct.pack(">I", len(json_bytes))
            writer.write(header + json_bytes)
            await writer.drain()
            logger.debug(f"Sent to Unity: {message}")
        except (ConnectionError, OSError) as e:
            logger.warning(f"Failed to send to Unity: {e}")
            async with self._command_lock:
                self._command_writer = None
                self.unity_connected = False

    # ─── Event Server (Unity → Python) ───────────────────────────────

    async def _handle_event_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Called when Unity connects to the event port (7778)."""
        addr = writer.get_extra_info("peername")
        logger.info(f"Unity connected to event port from {addr}")

        try:
            while True:
                # Read 4-byte length header
                header = await reader.readexactly(4)
                msg_len = struct.unpack(">I", header)[0]

                # Safety check: don't read absurdly large messages
                if msg_len > 1_000_000:
                    logger.error(f"Message too large ({msg_len} bytes), dropping connection")
                    break

                # Read the JSON body
                json_bytes = await reader.readexactly(msg_len)
                message = json.loads(json_bytes.decode("utf-8"))

                logger.debug(f"Received from Unity: {message}")

                # Dispatch to the callback
                if self._event_callback:
                    try:
                        await self._event_callback(message)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")

        except asyncio.IncompleteReadError:
            pass  # Unity disconnected mid-message
        except (asyncio.CancelledError, ConnectionError):
            pass
        except Exception as e:
            logger.error(f"Event client error: {e}")
        finally:
            logger.info("Unity disconnected from event port")
            writer.close()


class NullBridge:
    """No-op bridge for terminal-only mode (config.UNITY_ENABLED = False).

    Mirrors the UnityBridge async surface so every callsite in main.py / actions.py
    / tts.py works unchanged. All commands drop at DEBUG — the user-visible text
    (transcriptions, assistant speech) is already logged by main.py and tts.py,
    so echoing it again here would just duplicate every turn in the console.
    """

    def __init__(self):
        self.unity_connected = False

    async def start(self, event_callback=None):
        logger.info("Unity disabled — running in terminal-only mode (no TCP bridge).")

    async def stop(self):
        return

    async def send_command(self, action: str, **kwargs):
        logger.debug(f"[null-bridge] drop: {action} {kwargs}")

    async def send_thought(self, state: str, text: str = ""):
        logger.debug(f"[null-bridge] thought:{state} {text}")

    async def send_keyboard(self, visible: bool):
        return

    async def send_avatar_config(self):
        return
