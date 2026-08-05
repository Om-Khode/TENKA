"""Lets a handler request a graceful process exit after the current turn.

actions/ sits below main.py in TENKA's layering (actions/ -> main.py), so a
handler can't reach up and set main.py's own shutdown event directly
without inverting that dependency. This is the generic, bottom-layer
signal both sides can touch: a handler calls request() when it knows the
process must restart before anything else can safely run (e.g. after a
backup restore replaces the live database out from under the running
process); main.py's turn loop checks is_requested() right where it already
checks for the "shutdown" intent, and drives the same graceful-exit path.
"""

_requested = False


def request() -> None:
    """Ask the process to exit gracefully after finishing the current turn."""
    global _requested
    _requested = True


def is_requested() -> bool:
    return _requested


def _reset_for_testing() -> None:
    global _requested
    _requested = False
