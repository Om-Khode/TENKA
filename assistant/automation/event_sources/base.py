"""EventSource protocol for event-driven monitors."""
import os
from typing import Callable, Protocol, runtime_checkable


def normalize_process_name(value: str) -> str:
    """A process name reduced to what a filter and an event can be compared on.

    Strips any directory, drops a trailing executable extension, lowercases,
    and trims. `"C:/Windows/Notepad.exe"`, `"notepad.exe"` and `"Notepad"` all
    become `"notepad"`.

    **One function because two halves of the same contract disagreed.**
    `window.py` built `source_app` by stripping `.exe`, so a focus event
    reported `"Notepad"`. The parse prompt's examples are bare names, but the
    model wrote `source_filter="notepad.exe"`, and the matcher asked whether the
    filter was a substring of the event:

        'notepad.exe' in 'notepad'   ->  False

    The monitor was created, loaded, reported active, and could never fire. No
    error anywhere -- a filter that matches nothing looks exactly like an event
    that never happened.

    Normalising at the comparison rather than at the write, deliberately: the
    stored `source_filter` is what the operator asked for and stays readable,
    and a filter written by hand, by an older build, or by a model on a
    different day all land on the same comparison. Fixing only the prompt would
    have left every monitor already in the database broken.

    Also strips `.com` and `.bat`, since Windows process names carry those too
    and the same mismatch would apply.
    """
    if not value:
        return ""
    name = os.path.basename(value.strip()).lower()
    for suffix in (".exe", ".com", ".bat", ".cmd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@runtime_checkable
class EventSource(Protocol):
    """Interface every event source must satisfy."""

    name: str
    event_types: frozenset[str]

    def start(self, dispatch_fn: Callable[[dict], None], **kwargs) -> None: ...
    def stop(self) -> None: ...
