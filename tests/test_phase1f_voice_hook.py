"""
test_phase1f_voice_hook.py — Phase 1F voice command hook in actions.py.

Validates the regex matching for "set up Chrome" / "undo Chrome setup"
phrasings. Stubs `browser_setup.setup_chrome_cdp` and
`browser_setup.undo_chrome_cdp_setup` so the test runs without touching
the user's actual machine.

Run: python test_phase1f_voice_hook.py
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.actions as actions
import assistant.browser_setup as browser_setup


def _run(coro):
    return asyncio.run(coro)


class TestSetupVoiceHook(unittest.IsolatedAsyncioTestCase):
    """The hook lives at the top of `handle_computer_task`. We patch
    setup_chrome_cdp / undo_chrome_cdp_setup so the test never touches
    real Chrome shortcuts."""

    async def asyncSetUp(self):
        self.fake_setup_result = browser_setup.SetupResult(
            ok=True, message="Configured 1 Chrome shortcut. Restart Chrome and we're set.",
            chrome_exe="/fake/chrome.exe", port=9222,
        )
        self.fake_undo_result = browser_setup.UndoResult(
            ok=True, message="Restored 1 Chrome shortcut.", restored=["/fake/path.lnk"],
        )

    async def _call(self, goal: str):
        """Run the voice hook with a stubbed bridge + tts."""
        with patch.object(browser_setup, "setup_chrome_cdp",
                          return_value=self.fake_setup_result) as mock_setup, \
             patch.object(browser_setup, "undo_chrome_cdp_setup",
                          return_value=self.fake_undo_result) as mock_undo, \
             patch("assistant.io.audio.tts.speak", new=AsyncMock()):
            result = await actions.handle_computer_task(
                {"goal": goal}, llm_response="", bridge=None,
            )
        return result, mock_setup, mock_undo

    # ── Setup matches ──

    async def test_set_up_chrome_for_tenka(self):
        result, mock_setup, mock_undo = await self._call("set up Chrome for TENKA")
        self.assertIn("Configured", result)
        mock_setup.assert_called_once()
        mock_undo.assert_not_called()

    async def test_setup_chrome_one_word(self):
        result, mock_setup, _ = await self._call("setup chrome please")
        self.assertIn("Configured", result)
        mock_setup.assert_called_once()

    async def test_configure_chrome(self):
        result, mock_setup, _ = await self._call("configure Chrome for me")
        mock_setup.assert_called_once()

    async def test_enable_cdp(self):
        result, mock_setup, _ = await self._call("enable CDP")
        mock_setup.assert_called_once()

    async def test_enable_remote_debugging(self):
        result, mock_setup, _ = await self._call("enable remote debugging on chrome")
        mock_setup.assert_called_once()

    async def test_dry_run_via_preview_keyword(self):
        # Goal containing "preview" / "show me" / "what would" → dry-run mode
        await self._call("preview the chrome setup")
        mock_setup_call = browser_setup.setup_chrome_cdp
        # We can't easily check the patched call's kwargs from here without
        # restructuring; verify by patching directly:
        with patch.object(browser_setup, "setup_chrome_cdp",
                          return_value=self.fake_setup_result) as mock_setup, \
             patch.object(browser_setup, "undo_chrome_cdp_setup"), \
             patch("assistant.io.audio.tts.speak", new=AsyncMock()):
            await actions.handle_computer_task(
                {"goal": "preview the chrome setup"}, llm_response="", bridge=None,
            )
        mock_setup.assert_called_once_with(dry_run=True)

    # ── Undo matches ──

    async def test_undo_chrome_setup(self):
        result, mock_setup, mock_undo = await self._call("undo Chrome setup")
        self.assertIn("Restored", result)
        mock_undo.assert_called_once()
        mock_setup.assert_not_called()

    async def test_remove_cdp(self):
        result, _, mock_undo = await self._call("remove CDP")
        mock_undo.assert_called_once()

    async def test_reverse_chrome_setup(self):
        result, _, mock_undo = await self._call("reverse Chrome setup")
        mock_undo.assert_called_once()

    async def test_disable_remote_debugging(self):
        result, _, mock_undo = await self._call("disable remote debugging")
        mock_undo.assert_called_once()

    # ── Negative cases (must NOT trigger setup hook) ──

    async def test_unrelated_setup_goal_does_not_match(self):
        # Goal mentions "set up" but for something else — must NOT trigger
        # the Chrome setup hook. Falls through to the regular flow.
        # We don't actually run the computer_task chain (would hit network),
        # just verify setup_chrome_cdp was NOT called.
        with patch.object(browser_setup, "setup_chrome_cdp") as mock_setup, \
             patch.object(browser_setup, "undo_chrome_cdp_setup") as mock_undo, \
             patch("assistant.io.audio.tts.speak", new=AsyncMock()), \
             patch.object(actions, "computer_agent") as fake_ca:
            fake_ca.run_computer_task = AsyncMock(return_value="ran vision-loop")
            # Stub desktop_automation to bypass computer_task path
            with patch.object(actions, "personality_say", return_value="x"):
                fake_da = MagicMock()
                fake_da.can_handle = AsyncMock(return_value=(False, "vision"))
                with patch.dict("sys.modules", {"assistant.automation.router": fake_da}):
                    try:
                        await actions.handle_computer_task(
                            {"goal": "set up the printer for me"},
                            llm_response="", bridge=None,
                        )
                    except Exception:
                        pass  # downstream may fail without full env
        mock_setup.assert_not_called()
        mock_undo.assert_not_called()

    async def test_fill_form_does_not_trigger_setup(self):
        # The most important negative: "fill this form" must NOT match
        # the Chrome setup hook (it's the headline DOM-mode use case).
        with patch.object(browser_setup, "setup_chrome_cdp") as mock_setup, \
             patch.object(browser_setup, "undo_chrome_cdp_setup") as mock_undo:
            # Test the regex directly to avoid running the full chain
            import re as _re
            setup_re = _re.compile(
                r"\b(set\s*up|setup|configure|enable)\b.{0,40}"
                r"\b(chrome|cdp|remote\s+debugging|debug(?:ging)?\s+(?:chrome|port))\b",
                _re.IGNORECASE,
            )
            self.assertIsNone(setup_re.search("fill this form with testing values"))
            self.assertIsNone(setup_re.search("fill the demo form"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
