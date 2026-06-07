# assistant/io/overlay/__main__.py
"""overlay subprocess entry point.

Run as: python -m assistant.io.overlay
Flags:  --headless   (skip Tk window creation, echo parsed events to stderr)

Spawned by overlay_manager.py. Reads JSON-line events from stdin.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import threading

from .ipc import parse_event


def _run_headless() -> int:
    """Test mode: parse stdin lines, echo result to stderr, exit on EOF."""
    sys.stdin.reconfigure(line_buffering=True)
    for line in sys.stdin:
        result = parse_event(line)
        if result.ok:
            evt = result.event
            if evt.get("type") == "cmd" and evt.get("cmd") == "quit":
                print(f"[overlay] quit cmd received", file=sys.stderr, flush=True)
                return 0
            print(f"[overlay] parsed: {evt}", file=sys.stderr, flush=True)
        else:
            print(f"[overlay] parse error: {result.error}", file=sys.stderr, flush=True)
        if result.warning:
            print(f"[overlay] warning: {result.warning}", file=sys.stderr, flush=True)
    return 0


def _run_gui() -> int:
    """Real mode: spawn Tk root, create one StatusPill, read stdin in background."""
    import tkinter as tk
    if not hasattr(ctypes, "windll"):
        print("[overlay] ctypes.windll not available — overlay requires Windows", file=sys.stderr)
        return 1
    from .windows import StatusPill

    root = tk.Tk()
    root.withdraw()  # no main window
    pill = StatusPill(root)

    state = {
        "phase": "IDLE", "detail": "", "step": None, "tier": None, "running": True,
        "prev_substantive": False,  # was the last shown phase non-IDLE / non-LISTENING?
        "hide_job": None,           # tk.after id for delayed hide after DONE/STOPPED
        "stopped_armed": False,     # STOPPED was just shown — suppress the next Done flash
    }

    # Phases that, when followed by IDLE, should briefly show "Done" first.
    # SPEAKING is excluded — the audio itself is the acknowledgment, no need
    # for a redundant green check after every small-talk reply.
    # STOPPED is excluded — the red X already announced the cancellation.
    _NON_DONE_TRANSITIONS = frozenset({"IDLE", "LISTENING", "SPEAKING", "DONE", "STOPPED"})
    _DONE_HOLD_MS = 1200
    _STOPPED_HOLD_MS = 1200

    def _on_event(evt: dict) -> None:
        if evt.get("type") == "cmd" and evt.get("cmd") == "quit":
            state["running"] = False
            root.after(0, root.quit)
            return
        state["phase"] = evt["phase"]
        state["detail"] = evt.get("detail", "")
        raw_step = evt.get("step")
        state["step"] = tuple(raw_step) if raw_step else None
        state["tier"] = evt.get("tier")
        root.after(0, _apply_state)

    def _cancel_hide_job():
        job = state.get("hide_job")
        if job is not None:
            try:
                root.after_cancel(job)
            except Exception:
                pass
            state["hide_job"] = None

    def _hide_now():
        state["hide_job"] = None
        pill.hide()

    def _apply_state() -> None:
        phase = state["phase"]
        if phase == "IDLE":
            # If STOPPED just flashed, the pill is already holding the X —
            # the trailing IDLE from the handler's finally must NOT collapse
            # it early or replace it with Done. Leave the timer running.
            if state["stopped_armed"]:
                return
            # If we just finished a substantive task, flash Done for ~1.2s.
            if state["prev_substantive"]:
                state["prev_substantive"] = False
                _cancel_hide_job()
                pill.update_content("DONE", "", None, None)
                pill.show()
                state["hide_job"] = root.after(_DONE_HOLD_MS, _hide_now)
            else:
                _cancel_hide_job()
                pill.hide()
            return
        if phase == "STOPPED":
            # Show the red X for ~1.2s, then hide. Mark stopped_armed so the
            # handler's finally-IDLE doesn't pre-empt the flash or trigger Done.
            _cancel_hide_job()
            state["stopped_armed"] = True
            state["prev_substantive"] = False
            pill.update_content("STOPPED", "", None, None)
            pill.show()

            def _hide_stopped():
                state["hide_job"] = None
                state["stopped_armed"] = False
                pill.hide()
            state["hide_job"] = root.after(_STOPPED_HOLD_MS, _hide_stopped)
            return
        # Non-IDLE, non-STOPPED update: cancel any pending hide and render.
        _cancel_hide_job()
        state["stopped_armed"] = False
        if phase not in _NON_DONE_TRANSITIONS:
            state["prev_substantive"] = True
        pill.update_content(phase, state["detail"], state["step"], state["tier"])
        pill.show()

    def _stdin_reader() -> None:
        sys.stdin.reconfigure(line_buffering=True)
        for line in sys.stdin:
            result = parse_event(line)
            if result.ok:
                _on_event(result.event)
            else:
                print(f"[overlay] {result.error}", file=sys.stderr, flush=True)
        state["running"] = False
        root.after(0, root.quit)

    threading.Thread(target=_stdin_reader, daemon=True, name="overlay-stdin").start()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass  # parent TENKA was Ctrl+C'd; exit cleanly
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="parse stdin only, no GUI (for tests)")
    args = parser.parse_args()
    if args.headless:
        return _run_headless()
    return _run_gui()


if __name__ == "__main__":
    sys.exit(main())
