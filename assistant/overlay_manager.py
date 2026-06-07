# assistant/overlay_manager.py
"""overlay subprocess lifecycle.

Spawns `python -m assistant.io.overlay` and wires its stdin to the
StatusBroadcaster. Owns respawn-on-crash with a budget (3 in 60s) to
avoid runaway loops. After exhaustion, enters silent mode (IPC writes
become no-ops); TENKA keeps running.
"""
from __future__ import annotations

import atexit
import json
import logging
import subprocess
import sys
import time

from assistant.io.status_broadcaster import status

logger = logging.getLogger("overlay_manager")


def _pref_enabled() -> bool:
    """Reads overlay_enabled preference (default True)."""
    try:
        from assistant.storage.db import get_db
        from assistant.storage.repos.preference import PreferenceRepo
        db = get_db()
        if db is None:
            return True
        repo = PreferenceRepo(db)
        pref = repo.get_preference("overlay_enabled")
        val = pref["value"] if pref else None
        if val is None:
            return True
        return str(val).lower() not in ("false", "0", "no", "off")
    except Exception:
        return True  # fail-open


_RESPAWN_BUDGET = 3
_RESPAWN_WINDOW_SECS = 60.0
_STOP_GRACE_SECS = 2.0


class OverlayManager:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._respawn_attempts: list[float] = []
        self._silent_mode = False

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not _pref_enabled():
            logger.info("[overlay_mgr] overlay_enabled=false — not spawning")
            return
        if self._silent_mode:
            logger.info("[overlay_mgr] silent mode — not spawning")
            return
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "assistant.io.overlay"],
                stdin=subprocess.PIPE, stdout=sys.stderr, stderr=sys.stderr,
                text=True, bufsize=1,
            )
        except OSError as e:
            logger.error("[overlay_mgr] spawn failed: %s", e)
            return
        status.attach_ipc(self._proc.stdin)
        status.set_on_overlay_dead(self.respawn)
        atexit.register(self.stop)
        logger.info("[overlay_mgr] started overlay pid=%s", self._proc.pid)

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        status.detach_ipc()
        try:
            proc.stdin.write(json.dumps({"v": 1, "type": "cmd", "cmd": "quit"}) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=_STOP_GRACE_SECS)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        logger.info("[overlay_mgr] stopped")

    def respawn(self) -> None:
        if self._silent_mode:
            return
        now = time.time()
        self._respawn_attempts = [t for t in self._respawn_attempts if now - t < _RESPAWN_WINDOW_SECS]
        self._respawn_attempts.append(now)
        if len(self._respawn_attempts) > _RESPAWN_BUDGET:
            logger.error("[overlay_mgr] respawn budget exhausted (%d in %.0fs) — going silent",
                         _RESPAWN_BUDGET, _RESPAWN_WINDOW_SECS)
            self._silent_mode = True
            return
        logger.warning("[overlay_mgr] respawning overlay (attempt %d)", len(self._respawn_attempts))
        self.stop()
        self.start()


overlay_manager = OverlayManager()
