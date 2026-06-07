# tests/test_overlay_ipc_e2e.py
import json
import subprocess
import sys
import time
import pytest


def test_headless_overlay_parses_status_and_exits_on_eof():
    proc = subprocess.Popen(
        [sys.executable, "-m", "assistant.io.overlay", "--headless"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    payloads = [
        {"v": 1, "type": "status", "phase": "CLICKING", "detail": "Send",
         "cursor_follows": True, "ts": 1.0},
        {"v": 1, "type": "status", "phase": "IDLE", "detail": "",
         "cursor_follows": False, "ts": 2.0},
        {"v": 1, "type": "cmd", "cmd": "ping"},
    ]
    for p in payloads:
        proc.stdin.write(json.dumps(p) + "\n")
    proc.stdin.flush()
    proc.stdin.close()  # EOF
    try:
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("overlay did not exit on EOF within 5s")
    assert proc.returncode == 0
    # Headless mode echoes parsed events to stderr
    assert "CLICKING" in err
    assert "IDLE" in err
    assert "ping" in err


def test_headless_overlay_rejects_malformed_lines():
    proc = subprocess.Popen(
        [sys.executable, "-m", "assistant.io.overlay", "--headless"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    proc.stdin.write("not-json\n")
    proc.stdin.write(json.dumps({"v": 999, "type": "status", "phase": "IDLE"}) + "\n")
    proc.stdin.write(json.dumps({"v": 1, "type": "cmd", "cmd": "quit"}) + "\n")
    proc.stdin.flush()
    try:
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("overlay did not quit")
    assert proc.returncode == 0
    assert "invalid json" in err.lower() or "json" in err.lower()
    assert "version" in err.lower()
