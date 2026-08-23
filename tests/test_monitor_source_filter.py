"""A monitor whose filter matches nothing looks exactly like a quiet machine.

Live test, Part 3. The monitor was created, loaded, and reported active:

    [MONITORS] Created #2: 'Notepad Time Notifier' (window_focus)
    [event-monitor] Reloaded monitors (1 active)
    "Done. I'll watch for that — monitor 'Notepad Time Notifier' is active."

It could never fire. `source_filter` was `'notepad.exe'`, and
`event_sources/window.py` builds `source_app` by stripping the extension, so a
focus event reported `'notepad'`. The matcher asked:

    'notepad.exe' in 'notepad'   ->  False

Two halves of one contract, disagreeing, with nothing between them. The parse
prompt's examples are bare names, so the model was not following an instruction
when it wrote `.exe` -- it was filling a gap.

**No error anywhere.** That is the part worth keeping in mind: a filter that
matches nothing is indistinguishable from an event that never happened, so the
only symptom is silence, and silence is what a monitor looks like most of the
time anyway.

Fixed at the comparison rather than at the write, so filters already stored --
by hand, by an older build, or by a model on a different day -- start working
without a migration. Fixing the prompt alone would have left every existing
monitor broken.

Run with:  py -3.11 -m pytest tests/test_monitor_source_filter.py -v
"""
import pathlib
import sys
import time

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.automation.event_bus import check_dispatch  # noqa: E402
from assistant.automation.event_sources.base import (  # noqa: E402
    normalize_process_name,
)


def _monitor(source_filter, **kw):
    base = dict(
        event_type="window_focus", source_filter=source_filter,
        condition_mode="code", condition_expr=None, condition_prompt=None,
        cooldown_secs=5, enabled=1,
    )
    base.update(kw)
    return base


def _focus_event(app="notepad", title="notes.txt - Notepad"):
    return {"event_type": "window_focus", "source_app": app,
            "window_title": title}


# ─── the normaliser ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("notepad.exe", "notepad"),
    ("Notepad", "notepad"),
    ("NOTEPAD.EXE", "notepad"),
    ("C:/Windows/System32/Notepad.exe", "notepad"),
    (r"C:\Windows\System32\notepad.exe", "notepad"),
    ("  Spotify.exe  ", "spotify"),
    ("cmd.bat", "cmd"),
    ("", ""),
])
def test_a_process_name_reduces_to_one_form(raw, expected):
    assert normalize_process_name(raw) == expected


def test_the_extension_is_stripped_from_the_end_not_the_middle():
    """`.replace(".exe", "")` -- the original -- removes the substring wherever
    it appears, so a dotted name containing it mid-string is corrupted.

    `"exeter.exe"` does *not* distinguish the two (both give "exeter"), which is
    worth saying: the first version of this test used only that case and passed
    against the replace-anywhere mutation. `"foo.exe.bak"` separates them.
    """
    assert normalize_process_name("foo.exe.bak") == "foo.exe.bak"
    assert normalize_process_name("exeter.exe") == "exeter"
    assert normalize_process_name("myexeapp") == "myexeapp"


# ─── the observed failure ────────────────────────────────────────────────────

def test_the_reported_monitor_now_fires():
    """Verbatim from the live test: filter `notepad.exe`, event `notepad`."""
    assert check_dispatch(_monitor("notepad.exe"), _focus_event(),
                          now=time.time()) is True


@pytest.mark.parametrize("stored", [
    "notepad.exe", "notepad", "Notepad.EXE", "NOTEPAD",
    r"C:\Windows\notepad.exe",
])
def test_every_way_a_filter_might_be_written_matches(stored):
    """A model on a different day, an older build, or a person typing it. The
    stored value is left readable and the comparison absorbs the variation."""
    assert check_dispatch(_monitor(stored), _focus_event(),
                          now=time.time()) is True


# ─── and it still filters ────────────────────────────────────────────────────

@pytest.mark.parametrize("stored", ["chrome.exe", "chrome", "spotify"])
def test_a_filter_for_another_app_still_rejects(stored):
    """**The direction a normaliser breaks.** Something that normalised both
    sides to the same value, or compared them the wrong way round, would make
    every monitor fire on every window change -- and `code_executor` monitors
    run generated code, so a filter that matches everything is worse than one
    that matches nothing."""
    assert check_dispatch(_monitor(stored), _focus_event(),
                          now=time.time()) is False


def test_a_substring_filter_still_matches_a_longer_process_name():
    """The original intent, preserved: `source_filter` does substring matching
    so "chrome" catches "chrome" whatever the surrounding name is. What changed
    is only that an extension on either side no longer defeats it."""
    assert check_dispatch(_monitor("chrome"),
                          _focus_event(app="googlechrome"),
                          now=time.time()) is True


def test_no_filter_matches_everything():
    """Null `source_filter` means "fire on every matching event", per the parse
    prompt. A normaliser that turned None into a real comparison would silently
    stop those firing."""
    assert check_dispatch(_monitor(None), _focus_event(), now=time.time()) is True
    assert check_dispatch(_monitor(""), _focus_event(), now=time.time()) is True


def test_the_event_type_still_gates_first():
    """Normalising a filter must not let a media event through a window
    monitor."""
    media = {"event_type": "media_changed", "source_app": "notepad",
             "title": "x", "artist": "y"}
    assert check_dispatch(_monitor("notepad"), media, now=time.time()) is False


# ─── one definition, used by both halves ─────────────────────────────────────

def test_the_window_source_uses_the_shared_normaliser():
    """The defect was two implementations of "the process name". The source's
    ad-hoc strip is gone; if it comes back, the halves can disagree again and
    the only symptom will be silence."""
    src = (_ROOT / "assistant" / "automation" / "event_sources"
           / "window.py").read_text(encoding="utf-8")
    assert "normalize_process_name" in src, (
        "the window source no longer shares the normaliser"
    )
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    assert '.replace(".exe"' not in code, (
        "the ad-hoc extension strip is back alongside the shared one"
    )


def test_the_matcher_uses_the_shared_normaliser():
    src = (_ROOT / "assistant" / "automation" / "event_bus.py").read_text(
        encoding="utf-8")
    block = src[src.index("src_filter = monitor.get"):]
    block = block[:block.index("dedup_key")]
    assert "normalize_process_name" in block, (
        "the filter comparison is back to raw strings"
    )
