"""
test_browser_handle.py — Latch Task 13: which browser, and whether anyone can tell.

The interesting property is not "does it pick the extension when the extension
is there". It is what happens when it *isn't*, because that failure is silent:
the task still runs, it just runs against a browser with none of the user's
sessions, and the first symptom is a login wall on a site they are signed into.

So the downgrade is pinned in two ways — the handle says `bundled`, and exactly
one INFO line says why. A log that mentions it once at startup and never again
would leave "why did it use the bundled browser?" unanswerable for every task
after the first.

Run: py -3.11 -m pytest tests/test_browser_handle.py -v
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

import assistant.config as config  # noqa: E402
from assistant.automation.browser import handle as bh  # noqa: E402
from assistant.automation.browser.page_adapter import PageAdapter  # noqa: E402
from assistant.io.api import extension_ws as ews  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class _FakeConnection:
    """Enough of a `LatchConnection` to be handed to `ExtensionPage`."""

    async def call(self, method, params=None, *, timeout=30.0):
        return {}


@pytest.fixture(autouse=True)
def _no_connection():
    ews.reset_state_for_test()
    yield
    ews.reset_state_for_test()


@pytest.fixture()
def bundled_never_launches(monkeypatch):
    """Replace the bundled path so no test in this file can start a browser.

    `_bundled` calls `ensure_browser`, which launches real Chromium. A unit test
    has no business doing that, and a file that launched one per case would be
    slow enough that someone would stop running it.
    """
    launched: list[str] = []

    async def fake_bundled(reason: str):
        launched.append(reason)
        return bh.BrowserHandle(kind="bundled", page=_FakePage(), connection=None)

    monkeypatch.setattr(bh, "_bundled", fake_bundled)
    return launched


class _FakePage:
    url = ""

    async def evaluate(self, expression, arg=None):
        return {}

    def locator(self, selector):
        raise AssertionError("not used")


# ─── The extension is connected ──────────────────────────────────────────


def test_a_connected_extension_is_used(bundled_never_launches):
    ews.register(ews.LatchConnection(send_json=_noop, browser_name="firefox"))
    handle = _run(bh.get_browser_handle())
    assert handle.kind == "latch"
    assert handle.connection is not None
    assert bundled_never_launches == [], "it fell back with an extension connected"


def test_the_handle_carries_a_page_adapter(bundled_never_launches):
    ews.register(ews.LatchConnection(send_json=_noop))
    handle = _run(bh.get_browser_handle())
    assert isinstance(handle.page, PageAdapter)


def test_is_user_browser_distinguishes_the_two(bundled_never_launches):
    ews.register(ews.LatchConnection(send_json=_noop))
    assert _run(bh.get_browser_handle()).is_user_browser is True
    ews.reset_state_for_test()
    assert _run(bh.get_browser_handle()).is_user_browser is False


async def _noop(_frame):
    return None


# ─── The downgrade ───────────────────────────────────────────────────────


def test_with_nothing_connected_the_handle_is_bundled(bundled_never_launches):
    handle = _run(bh.get_browser_handle())
    assert handle.kind == "bundled"
    assert handle.connection is None


def test_the_downgrade_says_why(caplog, monkeypatch):
    """Exercises the REAL `_bundled`, because the log line lives there — but
    with the launch itself stubbed out.

    The first version of this test simply called `_bundled` and let it run. It
    passed, in twenty-seven seconds, having started an actual Chromium. A unit
    test has no business launching a browser: it is slow enough that someone
    stops running the file, and on this project it is the exact shape of thing
    that has twice reached the real desktop.
    """
    from assistant.automation.browser import automation as browser_automation

    async def refuse(*a, **k):
        raise RuntimeError("stubbed: no browser is launched in unit tests")

    monkeypatch.setattr(browser_automation, "ensure_browser", refuse)

    with caplog.at_level(logging.INFO, logger="browser.handle"):
        with pytest.raises(RuntimeError, match="stubbed"):
            _run(bh._bundled("no browser extension is connected"))

    lines = [r.message for r in caplog.records if "bundled" in r.message]
    assert len(lines) == 1, f"expected exactly one downgrade line, got {lines}"
    assert "no browser extension is connected" in lines[0], (
        "the log says it downgraded but not why. 'It used the wrong browser' "
        "is not a debuggable report."
    )


def test_every_downgrade_is_logged_not_just_the_first(monkeypatch, caplog):
    reasons: list[str] = []

    async def counting_bundled(reason: str):
        reasons.append(reason)
        return bh.BrowserHandle(kind="bundled", page=_FakePage())

    monkeypatch.setattr(bh, "_bundled", counting_bundled)
    for _ in range(3):
        _run(bh.get_browser_handle())
    assert len(reasons) == 3, (
        "a later resolution skipped the downgrade path. A run that mentions the "
        "fallback once and then goes quiet leaves every task after the first "
        "unexplained."
    )


# ─── prefer_latch=False ──────────────────────────────────────────────────


def test_prefer_latch_false_never_touches_the_connection(monkeypatch, bundled_never_launches):
    """Not merely 'loses the race' — never asks.

    A caller that asked for the bundled browser wants a clean profile. Handing
    it the user's live session because one happened to be available would be a
    different task than the one it asked for, and it would carry the user's
    cookies into it.
    """
    ews.register(ews.LatchConnection(send_json=_noop))
    asked = []
    monkeypatch.setattr(ews, "current_connection",
                        lambda: asked.append(True) or None)

    handle = _run(bh.get_browser_handle(prefer_latch=False))
    assert handle.kind == "bundled"
    assert asked == [], "it consulted the extension after being told not to"


def test_the_config_flag_off_also_never_touches_the_connection(monkeypatch, bundled_never_launches):
    ews.register(ews.LatchConnection(send_json=_noop))
    asked = []
    monkeypatch.setattr(ews, "current_connection",
                        lambda: asked.append(True) or None)
    monkeypatch.setattr(config, "BROWSER_PREFER_EXTENSION", False)

    handle = _run(bh.get_browser_handle())
    assert handle.kind == "bundled"
    assert asked == []


def test_the_reasons_are_distinguishable(monkeypatch):
    """An operator reading a log needs to tell 'nothing connected' from
    'you turned it off' — one is a setting, the other is a browser to open."""
    seen: list[str] = []

    async def capture(reason: str):
        seen.append(reason)
        return bh.BrowserHandle(kind="bundled", page=_FakePage())

    monkeypatch.setattr(bh, "_bundled", capture)

    _run(bh.get_browser_handle(prefer_latch=False))
    monkeypatch.setattr(config, "BROWSER_PREFER_EXTENSION", False)
    _run(bh.get_browser_handle())
    monkeypatch.setattr(config, "BROWSER_PREFER_EXTENSION", True)
    _run(bh.get_browser_handle())

    assert len(set(seen)) == 3, f"the three downgrade reasons are not distinct: {seen}"


# ─── A dropped connection ────────────────────────────────────────────────


def test_a_closed_connection_is_not_offered(bundled_never_launches):
    """A connection that died between check and use must surface as a
    downgrade, not as a handle whose every call raises."""
    conn = ews.LatchConnection(send_json=_noop)
    ews.register(conn)
    conn.close("socket dropped")

    handle = _run(bh.get_browser_handle())
    assert handle.kind == "bundled", (
        "a dead connection was handed out as a live browser. Every call on it "
        "would raise, one at a time, for the length of the task."
    )
