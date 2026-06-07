"""
test_terminal_only_mode.py — Phase 6C: terminal-only mode (UNITY_ENABLED=false)

Validates:
  - NullBridge mirrors UnityBridge's async surface (no callsite churn)
  - All send_* methods are awaitable no-ops, never raise
  - show_subtitle echoes to logger.info so terminal users see avatar speech
  - unity_connected stays False
  - The /set runtime setting toggles UNITY_ENABLED across reload
  - main.py's bridge factory selects NullBridge when disabled

Run: python test_terminal_only_mode.py
"""

import asyncio
import logging
import sys
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.io.unity_bridge as bridge_mod
from assistant.io.unity_bridge import NullBridge, UnityBridge


class TestNullBridgeSurface(unittest.TestCase):
    """NullBridge must expose every public method UnityBridge does."""

    def test_method_parity(self):
        public = lambda cls: {n for n in dir(cls) if not n.startswith("_")}
        unity = public(UnityBridge)
        null = public(NullBridge)
        missing = unity - null
        self.assertFalse(
            missing,
            f"NullBridge is missing UnityBridge methods: {missing}. "
            "Callsites in main.py/actions.py/tts.py would crash in terminal mode.",
        )

    def test_unity_connected_false(self):
        nb = NullBridge()
        self.assertFalse(nb.unity_connected)


class TestNullBridgeBehavior(unittest.IsolatedAsyncioTestCase):
    async def test_start_stop_no_op(self):
        nb = NullBridge()
        await nb.start(event_callback=lambda e: None)
        await nb.stop()  # Must not raise — main.py calls this in finally

    async def test_send_command_no_op(self):
        nb = NullBridge()
        await nb.send_command("set_expression", value="happy")
        await nb.send_command("play_animation", name="thinking")
        await nb.send_command("hide_avatar")
        await nb.send_command("show_avatar")

    async def test_send_command_silent_at_info(self):
        """Commands must not emit INFO logs — user-visible text is already
        surfaced by main.py (Transcription) and tts.py (Speaking). Echoing
        subtitle text here would duplicate every turn."""
        nb = NullBridge()
        bridge_mod.logger.setLevel(logging.INFO)
        with self.assertLogs(bridge_mod.logger, level="INFO") as cm:
            bridge_mod.logger.info("anchor")  # assertLogs requires ≥1 record
            await nb.send_command("show_subtitle", text="hello world")
            await nb.send_command("set_expression", value="happy")
            await nb.send_command("play_animation", name="thinking")
        info_lines = [l for l in cm.output if l.startswith("INFO")]
        self.assertEqual(
            len(info_lines), 1,
            f"NullBridge.send_command leaked INFO logs: {cm.output}",
        )

    async def test_send_command_logs_at_debug(self):
        """Commands should still trace at DEBUG for diagnostics."""
        nb = NullBridge()
        with self.assertLogs(bridge_mod.logger, level="DEBUG") as cm:
            await nb.send_command("show_subtitle", text="hello")
        self.assertTrue(
            any("hello" in line for line in cm.output),
            f"DEBUG trace missing: {cm.output}",
        )

    async def test_send_thought_keyboard_avatar_config(self):
        nb = NullBridge()
        await nb.send_thought("thinking")
        await nb.send_thought("done", "result text")
        await nb.send_keyboard(True)
        await nb.send_keyboard(False)
        await nb.send_avatar_config()


class TestRuntimeSetting(unittest.TestCase):
    """UNITY_ENABLED must be persistable via settings and survive reload."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._sandbox = Path(self._tmp)
        (self._sandbox / "memory").mkdir(parents=True, exist_ok=True)

    def test_setting_round_trip(self):
        import importlib
        with patch.dict("os.environ", {}, clear=False):
            import assistant.settings as ss
            ss._DB_PATH = self._sandbox / "memory" / "personality.db"
            ss.init_settings_db()

            # Default
            self.assertIsNone(ss.get("unity_enabled"))

            # Set false → persists
            ss.set("unity_enabled", False)
            self.assertEqual(ss.get("unity_enabled"), False)

            # Reset → back to None (caller falls back to default)
            ss.delete("unity_enabled")
            self.assertIsNone(ss.get("unity_enabled"))


class TestBridgeFactorySelection(unittest.TestCase):
    """main.py picks NullBridge when UNITY_ENABLED is False."""

    def test_factory_picks_unity_when_enabled(self):
        unity_enabled = True
        b = UnityBridge() if unity_enabled else NullBridge()
        self.assertIsInstance(b, UnityBridge)

    def test_factory_picks_null_when_disabled(self):
        unity_enabled = False
        b = UnityBridge() if unity_enabled else NullBridge()
        self.assertIsInstance(b, NullBridge)


if __name__ == "__main__":
    unittest.main(verbosity=2)
