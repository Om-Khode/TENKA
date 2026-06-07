"""
test_browser_setup.py — Phase 1F: Chrome CDP shortcut auto-setup.

The setup creates a user-owned `Chrome (TENKA-CDP).lnk` on the user's
Desktop and in their Start Menu (both user-writable — no admin required).
The user's existing Chrome shortcuts are NOT touched.

Coverage:
  - _ps_quote: single-quoted PowerShell literals; embedded apostrophes escape
  - _ensure_cdp_flag: idempotent / replace / append (kept — used by tests
    of the legacy helper, still part of the public-ish surface)
  - is_setup_done: marker missing → False; marker present + matching port
    + at least one created shortcut still on disk → True; corrupt marker
    → False; matching port but all shortcuts deleted → False
  - find_chrome_executable: registry + Program Files fallback (smoke import)
  - setup_chrome_cdp:
      - refuses when Chrome not found
      - returns ok=True with "already set up" when marker matches and a
        created shortcut still exists
      - happy path: creates shortcuts in user-writable targets, writes marker
      - dry_run reports without writing files
      - failure path when _create_lnk returns False everywhere
  - undo_chrome_cdp_setup: deletes created shortcuts, deletes marker, no-op
    when no marker, handles already-deleted shortcuts
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.setup as bs


# ─── _ps_quote ───────────────────────────────────────────────────────────


class TestPsQuote(unittest.TestCase):
    def test_simple_path(self):
        self.assertEqual(
            bs._ps_quote(r"C:\Users\Alice\Desktop\Chrome.lnk"),
            r"'C:\Users\Alice\Desktop\Chrome.lnk'",
        )

    def test_path_with_parentheses(self):
        path = r"C:\Users\bob\Desktop\Python 3.11 (64-bit).lnk"
        self.assertEqual(bs._ps_quote(path), f"'{path}'")

    def test_apostrophe_escaped(self):
        self.assertEqual(
            bs._ps_quote(r"C:\Users\O'Brien\Desktop\App.lnk"),
            r"'C:\Users\O''Brien\Desktop\App.lnk'",
        )

    def test_empty_string(self):
        self.assertEqual(bs._ps_quote(""), "''")


# ─── _ensure_cdp_flag (legacy helper, retained) ─────────────────────────


class TestEnsureCdpFlag(unittest.TestCase):
    def test_appends_when_absent(self):
        new, modified = bs._ensure_cdp_flag("--profile-directory=Default", 9222)
        self.assertEqual(new, "--profile-directory=Default --remote-debugging-port=9222")
        self.assertTrue(modified)

    def test_no_change_when_already_present(self):
        new, modified = bs._ensure_cdp_flag(
            "--profile-directory=Default --remote-debugging-port=9222", 9222,
        )
        self.assertFalse(modified)

    def test_replaces_wrong_port(self):
        new, modified = bs._ensure_cdp_flag(
            "--remote-debugging-port=9333 --other-flag", 9222,
        )
        self.assertTrue(modified)
        self.assertIn("--remote-debugging-port=9222", new)
        self.assertIn("--other-flag", new)
        self.assertNotIn("9333", new)

    def test_appends_when_args_empty(self):
        new, modified = bs._ensure_cdp_flag("", 9222)
        self.assertEqual(new, "--remote-debugging-port=9222")
        self.assertTrue(modified)

    def test_case_insensitive_match(self):
        _, modified = bs._ensure_cdp_flag("--Remote-Debugging-Port=9222", 9222)
        self.assertFalse(modified)


# ─── Marker file helpers ─────────────────────────────────────────────────


class TestMarker(unittest.TestCase):
    """is_setup_done now requires marker AND at least one referenced
    shortcut to still exist on disk."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tenka-test-")
        self._user_dir_patch = patch.object(
            bs, "_user_dir", return_value=Path(self._tmp),
        )
        self._user_dir_patch.start()
        self._sc_dir = Path(tempfile.mkdtemp(prefix="tenka-sc-"))

    def tearDown(self):
        self._user_dir_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._sc_dir, ignore_errors=True)

    def _write_marker_with_shortcut(self, port: int, sc_exists: bool):
        sc_path = self._sc_dir / "Chrome (TENKA-CDP).lnk"
        if sc_exists:
            sc_path.write_bytes(b"fake-lnk")
        marker = Path(self._tmp) / "chrome_cdp_setup.json"
        marker.write_text(json.dumps({
            "version": bs._SETUP_SCHEMA_VERSION,
            "port": port,
            "shortcuts_modified": [{
                "path": str(sc_path),
                "backup_path": "",
                "original_args": "",
                "new_args": f"--remote-debugging-port={port}",
            }],
        }))

    def test_setup_done_when_marker_missing(self):
        self.assertFalse(bs.is_setup_done())

    def test_setup_done_when_marker_present_and_shortcut_exists(self):
        self._write_marker_with_shortcut(9222, sc_exists=True)
        self.assertTrue(bs.is_setup_done(port=9222))

    def test_setup_done_false_when_shortcut_was_deleted(self):
        """User manually removed the .lnk — we must not claim setup is done."""
        self._write_marker_with_shortcut(9222, sc_exists=False)
        self.assertFalse(bs.is_setup_done(port=9222))

    def test_setup_done_false_for_different_port(self):
        self._write_marker_with_shortcut(9333, sc_exists=True)
        self.assertFalse(bs.is_setup_done(port=9222))

    def test_setup_done_false_for_corrupt_marker(self):
        (Path(self._tmp) / "chrome_cdp_setup.json").write_text("not valid json {")
        self.assertFalse(bs.is_setup_done())

    def test_setup_done_false_for_marker_with_no_shortcuts(self):
        (Path(self._tmp) / "chrome_cdp_setup.json").write_text(
            json.dumps({"port": 9222, "shortcuts_modified": []})
        )
        self.assertFalse(bs.is_setup_done(port=9222))

    def test_setup_done_false_when_schema_version_outdated(self):
        """Upgrade path: a marker from an older Phase 1F build has a stale
        schema version. is_setup_done must return False so setup_chrome_cdp
        rewrites the shortcut with the current args (e.g. --user-data-dir
        was added in v2 — without it, the shortcut silently merges into the
        user's main Chrome process)."""
        sc_path = self._sc_dir / "Chrome (TENKA-CDP).lnk"
        sc_path.write_bytes(b"fake-lnk")  # the old shortcut still exists
        (Path(self._tmp) / "chrome_cdp_setup.json").write_text(json.dumps({
            "version": bs._SETUP_SCHEMA_VERSION - 1,  # one version old
            "port": 9222,
            "shortcuts_modified": [{"path": str(sc_path)}],
        }))
        self.assertFalse(bs.is_setup_done(port=9222))

    def test_setup_done_writes_current_schema_version(self):
        """Marker writes must use _SETUP_SCHEMA_VERSION so future runs
        recognise them as current."""
        result = bs.SetupResult(
            ok=True, message="x",
            modified=[bs.ShortcutModification(path="/fake/sc.lnk", backup_path="")],
            chrome_exe="c.exe", port=9222,
        )
        bs._write_marker(result)
        data = json.loads(
            (Path(self._tmp) / "chrome_cdp_setup.json").read_text()
        )
        self.assertEqual(data["version"], bs._SETUP_SCHEMA_VERSION)

    def test_write_marker_creates_user_dir(self):
        Path(self._tmp).rmdir()
        result = bs.SetupResult(
            ok=True, message="x",
            modified=[bs.ShortcutModification(path="a", backup_path="")],
            chrome_exe="c.exe", port=9222,
        )
        bs._write_marker(result)
        marker = Path(self._tmp) / "chrome_cdp_setup.json"
        self.assertTrue(marker.is_file())
        data = json.loads(marker.read_text())
        self.assertEqual(data["port"], 9222)
        self.assertEqual(len(data["shortcuts_modified"]), 1)


# ─── setup_chrome_cdp ────────────────────────────────────────────────────


class TestSetupChromeCdp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tenka-test-")
        self._user_dir_patch = patch.object(
            bs, "_user_dir", return_value=Path(self._tmp),
        )
        self._user_dir_patch.start()
        # Sandbox shortcut targets so we don't write to the real Desktop
        self._sc_dir = Path(tempfile.mkdtemp(prefix="tenka-targets-"))
        target_a = self._sc_dir / "fake-desktop" / "Chrome (TENKA-CDP).lnk"
        target_b = self._sc_dir / "fake-startmenu" / "Chrome (TENKA-CDP).lnk"
        target_a.parent.mkdir(parents=True)
        target_b.parent.mkdir(parents=True)
        self._targets = [target_a, target_b]
        self._targets_patch = patch.object(
            bs, "_shortcut_targets", return_value=self._targets,
        )
        self._targets_patch.start()

    def tearDown(self):
        self._targets_patch.stop()
        self._user_dir_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._sc_dir, ignore_errors=True)

    def test_refuses_when_chrome_not_found(self):
        with patch.object(bs, "find_chrome_executable", return_value=None):
            r = bs.setup_chrome_cdp()
        self.assertFalse(r.ok)
        self.assertIn("install", r.message.lower())

    def test_already_configured_short_circuits(self):
        # Pre-write marker + a real .lnk file so is_setup_done returns True
        sc = self._targets[0]
        sc.write_bytes(b"fake")
        (Path(self._tmp) / "chrome_cdp_setup.json").write_text(json.dumps({
            "version": bs._SETUP_SCHEMA_VERSION,
            "port": 9222,
            "shortcuts_modified": [{"path": str(sc)}],
        }))
        with patch.object(bs, "find_chrome_executable",
                          return_value=Path("/fake/chrome.exe")):
            r = bs.setup_chrome_cdp(port=9222)
        self.assertTrue(r.ok)
        self.assertIn("already", r.message.lower())

    def test_happy_path_creates_shortcuts_and_writes_marker(self):
        with patch.object(bs, "find_chrome_executable",
                          return_value=Path("/fake/chrome.exe")), \
             patch.object(bs, "_create_lnk", return_value=True) as mock_create:
            r = bs.setup_chrome_cdp()

        self.assertTrue(r.ok)
        self.assertEqual(len(r.modified), 2)
        # Both targets attempted
        self.assertEqual(mock_create.call_count, 2)
        # Marker written
        self.assertTrue((Path(self._tmp) / "chrome_cdp_setup.json").is_file())
        # Message is short and TTS-friendly (no paths, no error codes)
        self.assertLess(len(r.message), 120, f"message too long for TTS: {r.message!r}")
        self.assertNotIn("\\", r.message)  # no Windows paths

    def test_shortcut_args_include_user_data_dir(self):
        """Critical correctness check: the created shortcut MUST pass
        --user-data-dir alongside --remote-debugging-port. Without
        --user-data-dir, opening the shortcut while Chrome is already
        running merges into the existing flag-less process and CDP is
        silently dropped — the original failure mode."""
        captured_args: list[str] = []

        def _capture(path, *, target, arguments, workdir, icon=""):
            captured_args.append(arguments)
            return True

        with patch.object(bs, "find_chrome_executable",
                          return_value=Path("/fake/chrome.exe")), \
             patch.object(bs, "_create_lnk", side_effect=_capture):
            r = bs.setup_chrome_cdp()

        self.assertTrue(r.ok)
        self.assertGreater(len(captured_args), 0)
        for args in captured_args:
            self.assertIn("--remote-debugging-port=9222", args)
            self.assertIn("--user-data-dir=", args)

    def test_dry_run_does_not_write_files(self):
        with patch.object(bs, "find_chrome_executable",
                          return_value=Path("/fake/chrome.exe")), \
             patch.object(bs, "_create_lnk") as mock_create:
            r = bs.setup_chrome_cdp(dry_run=True)

        self.assertTrue(r.ok)
        self.assertEqual(len(r.modified), 2)
        # _create_lnk never called in dry-run
        mock_create.assert_not_called()
        # No marker written
        self.assertFalse((Path(self._tmp) / "chrome_cdp_setup.json").is_file())

    def test_fails_cleanly_when_create_lnk_fails_everywhere(self):
        with patch.object(bs, "find_chrome_executable",
                          return_value=Path("/fake/chrome.exe")), \
             patch.object(bs, "_create_lnk", return_value=False):
            r = bs.setup_chrome_cdp()

        self.assertFalse(r.ok)
        self.assertEqual(len(r.modified), 0)
        self.assertEqual(len(r.skipped), 2)
        # No marker written on failure
        self.assertFalse((Path(self._tmp) / "chrome_cdp_setup.json").is_file())
        # Message is short, no paths, no PowerShell internals
        self.assertLess(len(r.message), 120)

    def test_partial_success_still_ok_and_marks(self):
        """If at least one shortcut was created, we count it as success."""
        # _create_lnk returns True for first target, False for second
        call_results = iter([True, False])
        with patch.object(bs, "find_chrome_executable",
                          return_value=Path("/fake/chrome.exe")), \
             patch.object(bs, "_create_lnk",
                          side_effect=lambda *a, **kw: next(call_results)):
            r = bs.setup_chrome_cdp()

        self.assertTrue(r.ok)
        self.assertEqual(len(r.modified), 1)
        self.assertEqual(len(r.skipped), 1)
        self.assertTrue((Path(self._tmp) / "chrome_cdp_setup.json").is_file())

    def test_no_targets_available(self):
        with patch.object(bs, "_shortcut_targets", return_value=[]), \
             patch.object(bs, "find_chrome_executable",
                          return_value=Path("/fake/chrome.exe")):
            r = bs.setup_chrome_cdp()

        self.assertFalse(r.ok)
        self.assertIn("desktop", r.message.lower())


# ─── undo_chrome_cdp_setup ───────────────────────────────────────────────


class TestUndoChromeCdpSetup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tenka-test-")
        self._user_dir_patch = patch.object(
            bs, "_user_dir", return_value=Path(self._tmp),
        )
        self._user_dir_patch.start()
        self._sc_dir = Path(tempfile.mkdtemp(prefix="tenka-sc-"))

    def tearDown(self):
        self._user_dir_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._sc_dir, ignore_errors=True)

    def test_no_op_when_marker_missing(self):
        r = bs.undo_chrome_cdp_setup()
        self.assertTrue(r.ok)
        self.assertIn("wasn't", r.message.lower())

    def test_deletes_shortcuts_and_marker(self):
        sc1 = self._sc_dir / "Chrome (TENKA-CDP).lnk"
        sc2 = self._sc_dir / "Chrome (TENKA-CDP) copy.lnk"
        sc1.write_bytes(b"fake")
        sc2.write_bytes(b"fake")
        (Path(self._tmp) / "chrome_cdp_setup.json").write_text(json.dumps({
            "port": 9222,
            "shortcuts_modified": [
                {"path": str(sc1)},
                {"path": str(sc2)},
            ],
        }))

        r = bs.undo_chrome_cdp_setup()

        self.assertTrue(r.ok)
        self.assertFalse(sc1.exists())
        self.assertFalse(sc2.exists())
        self.assertFalse((Path(self._tmp) / "chrome_cdp_setup.json").is_file())
        self.assertEqual(len(r.restored), 2)

    def test_handles_already_deleted_shortcut(self):
        """If the user manually deleted the .lnk, undo should still succeed."""
        sc = self._sc_dir / "Chrome (TENKA-CDP).lnk"
        # Don't write the file — simulate user already deleted it
        (Path(self._tmp) / "chrome_cdp_setup.json").write_text(json.dumps({
            "port": 9222,
            "shortcuts_modified": [{"path": str(sc)}],
        }))

        r = bs.undo_chrome_cdp_setup()

        self.assertTrue(r.ok)
        # Counted as success since end-state matches
        self.assertEqual(len(r.restored), 1)
        self.assertEqual(len(r.failed), 0)
        # Marker still cleaned up
        self.assertFalse((Path(self._tmp) / "chrome_cdp_setup.json").is_file())


# ─── _shortcut_targets ───────────────────────────────────────────────────


class TestShortcutTargets(unittest.TestCase):
    """The targets list should only include locations whose parent dir
    exists. We don't want to try to write to a non-existent Desktop folder."""

    def test_returns_list(self):
        # Smoke test — just ensure it doesn't crash and returns a list
        targets = bs._shortcut_targets()
        self.assertIsInstance(targets, list)
        # Each path should end with the well-known shortcut name
        for t in targets:
            self.assertEqual(t.name, "Chrome (TENKA-CDP).lnk")


if __name__ == "__main__":
    unittest.main(verbosity=2)
